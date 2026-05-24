"""
Process Real Human EEG Data
Processes the downloaded EEGLAB format data from OpenNeuro ds004306.
"""

import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def process_real_human_eeg():
    """
    Process the real human EEG data from sub-03.
    """
    print("=" * 70)
    print("PROCESSING REAL HUMAN EEG DATA")
    print("=" * 70)
    
    # File paths
    eeg_file = Path("data/sub-03/ses-03/eeg/sub-03_ses-03_task-experiment_run-01_eeg.set")
    events_file = Path("data/sub-03/ses-03/eeg/sub-03_ses-03_task-experiment_run-01_events.tsv")
    
    # Load EEGLAB file
    print("\nLoading EEGLAB file...")
    try:
        raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False, montage_units='mm')
    except FileNotFoundError:
        print("Error: .fdt file not found. The .set file requires a companion .fdt data file.")
        print("This dataset may need to be downloaded completely or converted to a different format.")
        print("Let me try downloading the complete dataset again...")
        
        import openneuro
        from openneuro import download
        
        # Download complete dataset without include filter
        download(
            dataset="ds004306",
            target_dir="data",
            tag="1.0.0"
        )
        
        # Try loading again
        raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False, montage_units='mm')
    print(f"Loaded: {len(raw.ch_names)} channels, {raw.n_times} timepoints")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Channels: {raw.ch_names[:10]}...")  # Show first 10 channels
    
    # Apply bandpass filter (0.1 - 20 Hz for smooth data)
    print("\nApplying bandpass filter (0.1 - 20 Hz)...")
    raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=20, verbose=False)
    print("Filter applied")
    
    # Load events from TSV file
    print("\nLoading events from TSV file...")
    events_df = pd.read_csv(events_file, sep='\t')
    print(f"Loaded {len(events_df)} events")
    print(f"Columns: {events_df.columns.tolist()}")
    print(f"\nFirst few events:")
    print(events_df.head())
    
    # Check for trial_type or value column
    if 'trial_type' in events_df.columns:
        print(f"\nTrial types found: {events_df['trial_type'].value_counts()}")
    elif 'value' in events_df.columns:
        print(f"\nEvent values found: {events_df['value'].value_counts()}")
    
    # Create event array for MNE
    # Convert onset (seconds) to sample indices
    sfreq = raw_filtered.info['sfreq']
    events_df['sample'] = (events_df['onset'] * sfreq).astype(int)
    
    # Create event IDs based on trial types
    # We'll assign different event IDs for different conditions
    unique_types = events_df['trial_type'].unique() if 'trial_type' in events_df.columns else events_df['value'].unique()
    event_id_map = {str(i+1): t for i, t in enumerate(unique_types)}
    reverse_event_id_map = {t: i+1 for i, t in enumerate(unique_types)}
    
    events_df['event_id'] = events_df['trial_type'].map(reverse_event_id_map) if 'trial_type' in events_df.columns else events_df['value'].map(reverse_event_id_map)
    
    # Create MNE events array
    events_array = []
    for idx, row in events_df.iterrows():
        events_array.append([row['sample'], 0, row['event_id']])
    events_array = np.array(events_array, dtype=int)
    
    print(f"\nCreated events array: {events_array.shape}")
    print(f"Event ID mapping: {event_id_map}")
    
    # Create epochs
    print("\nCreating epochs with baseline correction...")
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
    
    # Explicitly apply baseline correction again
    epochs.apply_baseline((-0.2, 0))
    
    print(f"Created epochs: {len(epochs)} trials")
    print(f"Epoch shape: {epochs.get_data().shape} (trials x channels x timepoints)")
    
    # Compute evoked responses for each condition
    print("\nComputing evoked responses...")
    evokeds = {}
    for event_name in event_id_map.values():
        if event_name in epochs.event_id:
            evoked = epochs[event_name].average()
            evoked.apply_baseline((-0.2, 0))
            evokeds[event_name] = evoked
            print(f"  {event_name}: {len(epochs[event_name])} trials")
    
    # Plot ERP comparison
    print("\nGenerating ERP comparison plot...")
    
    # Get available channels
    all_channels = list(evokeds.values())[0].ch_names
    
    # Try to find central/parietal channels
    preferred_channels = ['Cz', 'Pz', 'P3', 'P4', 'Fz', 'Fp1', 'Fp2']
    available_channels = [ch for ch in preferred_channels if ch in all_channels]
    
    if not available_channels:
        # Use first 4 channels as fallback
        available_channels = all_channels[:4]
        print(f"Using first {len(available_channels)} available channels")
    else:
        print(f"Plotting channels: {available_channels}")
    
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
    print("Saved ERP plot to: real_human_erp.png")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Conditions processed: {len(evokeds)}")
    print(f"Total trials: {len(epochs)}")
    print(f"Channels: {len(all_channels)}")
    print(f"Time window: -0.2 to 0.8 seconds")
    print(f"Filter: 0.1 - 20 Hz bandpass")
    print("\nOutput file:")
    print("  - real_human_erp.png")

if __name__ == "__main__":
    process_real_human_eeg()
