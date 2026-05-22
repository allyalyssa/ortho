"""
Process Brainwaves with MNE-Python
Loads raw EEG data, applies filtering, creates epochs around word stimuli,
and computes ERPs for high vs low interference conditions.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_raw_eeg():
    """
    Load the raw EEG data file.
    """
    print("Loading raw EEG data...")
    
    # Find the EEG file
    eeg_dir = Path("data/sub-01/eeg")
    eeg_files = list(eeg_dir.glob("*.fif"))
    
    if not eeg_files:
        print("No .fif file found. Looking for other formats...")
        eeg_files = list(eeg_dir.glob("*.set"))
        if not eeg_files:
            eeg_files = list(eeg_dir.glob("*.edf"))
            if not eeg_files:
                raise FileNotFoundError("No EEG data file found in data/sub-01/eeg/")
    
    eeg_file = eeg_files[0]
    print(f"Found EEG file: {eeg_file}")
    
    # Load the raw data
    raw = mne.io.read_raw_fif(eeg_file, preload=True)
    
    print(f"Loaded: {len(raw.ch_names)} channels, {raw.n_times} timepoints")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    
    return raw

def apply_filter(raw):
    """
    Apply bandpass filter (0.1 - 30 Hz) to remove slow drift and high-frequency noise.
    """
    print("\nApplying bandpass filter (0.1 - 30 Hz)...")
    raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=30, verbose=False)
    print("Filter applied successfully")
    return raw_filtered

def load_tagged_events():
    """
    Load the tagged events CSV file.
    """
    print("\nLoading tagged events...")
    events_df = pd.read_csv('tagged_events.csv')
    print(f"Loaded {len(events_df)} events")
    print(f"Columns: {events_df.columns.tolist()}")
    
    # Filter out trials without interference level
    events_df = events_df[events_df['interference_level'].notna()]
    print(f"Trials with interference tags: {len(events_df)}")
    
    return events_df

def create_epochs(raw, events_df):
    """
    Create epochs around word stimuli (-0.2 to 0.8 seconds).
    """
    print("\nCreating epochs...")
    
    # Convert onset (ms) to sample indices
    sfreq = raw.info['sfreq']
    events_df['sample'] = (events_df['onset'] / 1000 * sfreq).astype(int)
    
    # Create MNE events array: [sample, 0, event_id]
    # We'll use different event IDs for high vs low interference
    event_id_map = {'high': 1, 'low': 2}
    events_df['event_id'] = events_df['interference_level'].map(event_id_map)
    
    # Create events array
    events_array = []
    for idx, row in events_df.iterrows():
        events_array.append([row['sample'], 0, row['event_id']])
    
    events_array = np.array(events_array, dtype=int)
    print(f"Created events array: {events_array.shape}")
    
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
    
    # Explicitly apply baseline correction
    epochs.apply_baseline((-0.2, 0))
    
    print(f"Created epochs: {len(epochs)} trials")
    print(f"Epoch shape: {epochs.get_data().shape} (trials x channels x timepoints)")
    
    return epochs

def compute_and_plot_erps(epochs):
    print("\nComputing ERPs...")
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
        print(f"Topomap error: {e}")
        
    return evoked_high, evoked_low

def main():
    """
    Main processing pipeline.
    """
    print("=" * 70)
    print("EEG PROCESSING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load raw EEG
    raw = load_raw_eeg()
    
    # Step 2: Apply filter
    raw_filtered = apply_filter(raw)
    
    # Step 3: Load tagged events
    events_df = load_tagged_events()
    
    # Step 4: Create epochs
    epochs = create_epochs(raw_filtered, events_df)
    
    # Step 5: Compute and plot ERPs
    evoked_high, evoked_low = compute_and_plot_erps(epochs)
    
    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nTotal epochs created: {len(epochs)}")
    print(f"High interference trials: {len(epochs['high_interference'])}")
    print(f"Low interference trials: {len(epochs['low_interference'])}")
    print(f"Time window: -0.2 to 0.8 seconds")
    print(f"Filter: 0.1 - 30 Hz bandpass")
    print("\nOutput files:")
    print("  - erp_comparison.png: ERP waveform comparison")
    print("  - erp_topomap_n400.png: Topographic map of N400 difference")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
