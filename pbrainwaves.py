"""
Process Brainwaves with MNE-Python
Loads raw EEG data, applies filtering, creates epochs around word stimuli,
and computes ERPs for high vs low interference conditions.
"""

import logging
import mne
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def load_raw_eeg() -> mne.io.Raw:
    """Load the raw EEG data file."""
    logger.info("Loading raw EEG data...")
    
    eeg_dir = Path("data/sub-01/eeg")
    eeg_files = list(eeg_dir.glob("*.fif"))
    
    if not eeg_files:
        logger.warning("No .fif file found. Looking for other formats...")
        eeg_files = list(eeg_dir.glob("*.set"))
        if not eeg_files:
            eeg_files = list(eeg_dir.glob("*.edf"))
            if not eeg_files:
                raise FileNotFoundError("No EEG data file found in data/sub-01/eeg/")
    
    eeg_file = eeg_files[0]
    logger.info(f"Found EEG file: {eeg_file}")
    
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    logger.info(f"Loaded: {len(raw.ch_names)} channels, {raw.n_times} timepoints")
    logger.info(f"Sampling frequency: {raw.info['sfreq']} Hz")
    logger.info(f"Duration: {raw.times[-1]:.1f} seconds")
    
    return raw

def apply_filter(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply bandpass filter (0.1 - 30 Hz) to remove slow drift and high-frequency noise."""
    logger.info("Applying bandpass filter (0.1 - 30 Hz)...")
    raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=30, verbose=False)
    logger.info("Filter applied successfully")
    return raw_filtered

def load_tagged_events() -> pd.DataFrame:
    """Load the tagged events CSV file."""
    logger.info("Loading tagged events...")
    events_df = pd.read_csv('tagged_events.csv')
    logger.info(f"Loaded {len(events_df)} events")
    logger.info(f"Columns: {events_df.columns.tolist()}")
    
    events_df = events_df[events_df['interference_level'].notna()]
    logger.info(f"Trials with interference tags: {len(events_df)}")
    
    return events_df

def create_epochs(raw: mne.io.Raw, events_df: pd.DataFrame) -> mne.BaseEpochs:
    """Create epochs around word stimuli (-0.2 to 0.8 seconds)."""
    logger.info("Creating epochs...")
    
    sfreq = raw.info['sfreq']
    events_df['sample'] = (events_df['onset'] / 1000 * sfreq).astype(int)
    
    event_id_map = {'high': 1, 'low': 2}
    events_df['event_id'] = events_df['interference_level'].map(event_id_map)
    
    # Create events array
    events_array = []
    for idx, row in events_df.iterrows():
        events_array.append([row['sample'], 0, row['event_id']])
    
    events_array = np.array(events_array, dtype=int)
    logger.info(f"Created events array: {events_array.shape}")
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events_array,
        event_id={'high_interference': 1, 'low_interference': 2},
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0),
        preload=True,
        verbose=False
    )
    
    epochs.apply_baseline((-0.2, 0))
    logger.info(f"Created epochs: {len(epochs)} trials")
    logger.info(f"Epoch shape: {epochs.get_data().shape} (trials x channels x timepoints)")
    
    return epochs

def compute_and_plot_erps(epochs: mne.BaseEpochs) -> tuple[mne.Evoked, mne.Evoked]:
    """Compute and plot ERPs for high vs low interference conditions."""
    logger.info("Computing ERPs...")
    evoked_high = epochs['high_interference'].average()
    evoked_low = epochs['low_interference'].average()
    
    evoked_high.apply_baseline((-0.2, 0))
    evoked_low.apply_baseline((-0.2, 0))
    
    channels_to_plot = ['Cz', 'Pz', 'P3', 'P4']
    available_channels = [ch for ch in channels_to_plot if ch in epochs.ch_names]
    
    fig, axes = plt.subplots(len(available_channels), 1, figsize=(10, 4 * len(available_channels)))
    if len(available_channels) == 1:
        axes = [axes]
        
    for idx, ch in enumerate(available_channels):
        ax = axes[idx]
        ch_idx = epochs.ch_names.index(ch)
        
        times = evoked_high.times * 1000
        high_data_uv = evoked_high.data[ch_idx, :] * 1e6
        low_data_uv = evoked_low.data[ch_idx, :] * 1e6
        
        ax.plot(times, high_data_uv, label='High Interference', color='red', linewidth=2)
        ax.plot(times, low_data_uv, label='Low Interference', color='blue', linewidth=2)
        ax.axvspan(300, 500, alpha=0.2, color='yellow', label='N400 Window')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (uV)')
        ax.set_title(f'ERP at {ch}: High vs Low Interference')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('erp_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    try:
        evoked_diff = mne.combine_evoked([evoked_high, evoked_low], weights=[1, -1])
        evoked_diff.apply_baseline((-0.2, 0))
        fig_topo = evoked_diff.plot_topomap(
            times=[0.4], ch_type='eeg', time_unit='s', show=False, scalings={'eeg': 1e6}
        )
        fig_topo.suptitle('N400 Difference (High - Low Interference) at 400ms')
        plt.savefig('erp_topomap_n400.png', dpi=150, bbox_inches='tight')
        plt.close(fig_topo)
    except Exception as e:
        logger.error(f"Topomap error: {e}")
        
    return evoked_high, evoked_low

def main() -> None:
    """Main processing pipeline."""
    logging.basicConfig(level=logging.INFO)
    raw = load_raw_eeg()
    
    raw_filtered = apply_filter(raw)
    
    events_df = load_tagged_events()
    
    epochs = create_epochs(raw_filtered, events_df)
    
    evoked_high, evoked_low = compute_and_plot_erps(epochs)
    
    logger.info(f"Total epochs created: {len(epochs)}")
    logger.info(f"High interference trials: {len(epochs['high_interference'])}")
    logger.info(f"Low interference trials: {len(epochs['low_interference'])}")
    logger.info(f"Time window: -0.2 to 0.8 seconds")
    logger.info(f"Filter: 0.1 - 30 Hz bandpass")
    logger.info("Output files: erp_comparison.png, erp_topomap_n400.png")

if __name__ == "__main__":
    main()
