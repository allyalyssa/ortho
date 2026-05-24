"""
Process Real Human EEG Data
Processes the downloaded EEGLAB format data from OpenNeuro ds004306.
"""

import logging
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

def process_real_human_eeg() -> None:
    """Process the real human EEG data from sub-03."""
    logging.basicConfig(level=logging.INFO)
    
    eeg_file = Path("data/sub-03/ses-03/eeg/sub-03_ses-03_task-experiment_run-01_eeg.set")
    events_file = Path("data/sub-03/ses-03/eeg/sub-03_ses-03_task-experiment_run-01_events.tsv")
    
    logger.info("Loading EEGLAB file...")
    try:
        raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False, montage_units='mm')
    except FileNotFoundError:
        logger.error(".fdt file not found. The .set file requires a companion .fdt data file.")
        logger.info("This dataset may need to be downloaded completely or converted to a different format.")
        logger.info("Trying to download the complete dataset again...")
        
        import openneuro
        from openneuro import download
        
        download(
            dataset="ds004306",
            target_dir="data",
            tag="1.0.0"
        )
        
        raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False, montage_units='mm')
    logger.info(f"Loaded: {len(raw.ch_names)} channels, {raw.n_times} timepoints")
    logger.info(f"Sampling frequency: {raw.info['sfreq']} Hz")
    logger.info(f"Channels: {raw.ch_names[:10]}...")
    
    logger.info("Applying bandpass filter (0.1 - 20 Hz)...")
    raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=20, verbose=False)
    logger.info("Filter applied")
    
    logger.info("Loading events from TSV file...")
    events_df = pd.read_csv(events_file, sep='\t')
    logger.info(f"Loaded {len(events_df)} events")
    logger.info(f"Columns: {events_df.columns.tolist()}")
    logger.info(f"First few events:\n{events_df.head()}")
    
    if 'trial_type' in events_df.columns:
        logger.info(f"Trial types found: {events_df['trial_type'].value_counts()}")
    elif 'value' in events_df.columns:
        logger.info(f"Event values found: {events_df['value'].value_counts()}")
    
    sfreq = raw_filtered.info['sfreq']
    events_df['sample'] = (events_df['onset'] * sfreq).astype(int)
    
    unique_types = events_df['trial_type'].unique() if 'trial_type' in events_df.columns else events_df['value'].unique()
    event_id_map = {str(i+1): t for i, t in enumerate(unique_types)}
    reverse_event_id_map = {t: i+1 for i, t in enumerate(unique_types)}
    
    events_df['event_id'] = events_df['trial_type'].map(reverse_event_id_map) if 'trial_type' in events_df.columns else events_df['value'].map(reverse_event_id_map)
    
    # Create MNE events array
    events_array = []
    for idx, row in events_df.iterrows():
        events_array.append([row['sample'], 0, row['event_id']])
    events_array = np.array(events_array, dtype=int)
    logger.info(f"Created events array: {events_array.shape}")
    logger.info(f"Event ID mapping: {event_id_map}")
    
    logger.info("Creating epochs with baseline correction...")
    epochs = mne.Epochs(
        raw_filtered,
        events_array,
        event_id=event_id_map,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0),  # Explicit baseline correction
        preload=True,
        verbose=False
    )
    
    epochs.apply_baseline((-0.2, 0))
    logger.info(f"Created epochs: {len(epochs)} trials")
    logger.info(f"Epoch shape: {epochs.get_data().shape} (trials x channels x timepoints)")
    
    logger.info("Computing evoked responses...")
    evokeds = {}
    for event_name in event_id_map.values():
        if event_name in epochs.event_id:
            evoked = epochs[event_name].average()
            evoked.apply_baseline((-0.2, 0))
            evokeds[event_name] = evoked
            logger.info(f"  {event_name}: {len(epochs[event_name])} trials")
    
    logger.info("Generating ERP comparison plot...")
    
    all_channels = list(evokeds.values())[0].ch_names
    
    preferred_channels = ['Cz', 'Pz', 'P3', 'P4', 'Fz', 'Fp1', 'Fp2']
    available_channels = [ch for ch in preferred_channels if ch in all_channels]
    
    if not available_channels:
        available_channels = all_channels[:4]
        logger.warning(f"Using first {len(available_channels)} available channels")
    else:
        logger.info(f"Plotting channels: {available_channels}")
    
    # Create comparison plot
    fig, axes = plt.subplots(len(available_channels), 1, figsize=(10, 4 * len(available_channels)))
    if len(available_channels) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(evokeds)))
    
    for idx, ch in enumerate(available_channels):
        ax = axes[idx]
        
        for (event_name, evoked), color in zip(evokeds.items(), colors):
            ch_idx = evoked.ch_names.index(ch)
            times = evoked.times * 1000
            data_uv = evoked.data[ch_idx, :] * 1e6
            ax.plot(times, data_uv, label=event_name, color=color, linewidth=2)
        
        ax.axvspan(300, 500, alpha=0.2, color='yellow', label='N400 Window')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (uV)')
        ax.set_title(f'ERP at {ch}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('real_human_erp.png', dpi=150, bbox_inches='tight')
    logger.info("Saved ERP plot to: real_human_erp.png")
    plt.close()
    
    logger.info(f"Conditions processed: {len(evokeds)}")
    logger.info(f"Total trials: {len(epochs)}")
    logger.info(f"Channels: {len(all_channels)}")
    logger.info(f"Time window: -0.2 to 0.8 seconds")
    logger.info(f"Filter: 0.1 - 20 Hz bandpass")
    logger.info("Output file: real_human_erp.png")

if __name__ == "__main__":
    process_real_human_eeg()
