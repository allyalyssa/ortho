"""
Grand Average Pipeline
Processes multiple subjects/runs and computes group-level ERPs for high vs low interference conditions.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def find_eeg_files():
    """
    Scan the data directory to find all EEG files.
    """
    print("Scanning data directory for EEG files...")
    data_dir = Path("./data/derivatives/preprocessed")
    
    # Find all .fif files in the preprocessed directory
    eeg_files = list(data_dir.rglob("*_ica_eeg.fif"))
    
    if not eeg_files:
        print("No EEG files found in data directory")
        return []
    
    print(f"Found {len(eeg_files)} EEG files:")
    for f in eeg_files:
        print(f"  - {f.name}")
    
    return eeg_files

def load_tagged_events():
    """
    Load the tagged events CSV if it exists.
    """
    tagged_file = Path("tagged_events.csv")
    if tagged_file.exists():
        df = pd.read_csv(tagged_file)
        print(f"Loaded tagged events: {len(df)} trials")
        return df
    return None

def process_single_subject(eeg_file, tagged_events=None):
    """
    Process a single subject's EEG data and return evoked responses.
    """
    print(f"\nProcessing: {eeg_file.name}")
    
    try:
        # Load raw data
        print("  Loading raw data...")
        raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
        print(f"  Loaded: {len(raw.ch_names)} channels, {raw.n_times} timepoints")
    except (ValueError, RuntimeError, Exception) as e:
        print(f"  Skipping corrupted file: {e}")
        return None, None
    
    # Map generic channel names to standard 10-20 system for MNE sample dataset
    channel_mapping = {
        'EEG 001': 'Fp1', 'EEG 002': 'Fp2', 'EEG 003': 'F3', 'EEG 004': 'F4',
        'EEG 005': 'C3', 'EEG 006': 'Cz', 'EEG 007': 'C4', 'EEG 008': 'P3',
        'EEG 009': 'Pz', 'EEG 010': 'P4', 'EEG 011': 'O1', 'EEG 012': 'O2',
        'EEG 013': 'F7', 'EEG 014': 'F8', 'EEG 015': 'T7', 'EEG 016': 'T8',
        'EEG 017': 'P7', 'EEG 018': 'P8', 'EEG 019': 'Fz', 'EEG 020': 'FCz'
    }
    
    # Rename channels if they match the mapping
    current_names = raw.ch_names
    new_names = [channel_mapping.get(ch, ch) for ch in current_names]
    if new_names != current_names:
        raw.rename_channels(dict(zip(current_names, new_names)))
        print(f"  Mapped {len([c for c in current_names if c in channel_mapping])} channels to 10-20 system")
    
    # Apply bandpass filter (narrower range for smoother data)
    print("  Applying bandpass filter (0.1 - 20 Hz)...")
    raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=20, verbose=False)
    
    # Load events
    if tagged_events is not None:
        # Use tagged events from CSV
        print("  Using tagged events from CSV...")
        events_df = tagged_events.copy()
        events_df = events_df[events_df['interference_level'].notna()]
        
        # Convert onset to sample indices (onset is in milliseconds, convert to seconds)
        sfreq = raw_filtered.info['sfreq']
        events_df['sample'] = (events_df['onset'] / 1000 * sfreq).astype(int)
        
        # Create event array - map 'high'/'low' to event IDs
        event_id_map = {'high': 1, 'low': 2}
        events_df['event_id'] = events_df['interference_level'].map(event_id_map)
        
        events_array = []
        for idx, row in events_df.iterrows():
            events_array.append([row['sample'], 0, row['event_id']])
        events_array = np.array(events_array, dtype=int)
        
        event_id_dict = {'high_interference': 1, 'low_interference': 2}
    else:
        # Use events from the FIF file
        print("  Using events from FIF file...")
        events_array = mne.find_events(raw_filtered, stim_channel='STI 014', verbose=False)
        event_id_dict = {'high_interference': 1, 'low_interference': 2}  # Placeholder
    
    # Create epochs with explicit baseline correction
    print("  Creating epochs with baseline correction...")
    epochs = mne.Epochs(
        raw_filtered,
        events_array,
        event_id=event_id_dict,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0),  # Explicit baseline correction
        preload=True,
        verbose=False
    )
    
    # Explicitly apply baseline correction again to ensure it's properly zeroed
    epochs.apply_baseline((-0.2, 0))
    
    print(f"  Created epochs: {len(epochs)} trials")
    
    # Compute evoked responses
    print("  Computing evoked responses...")
    
    # Check if we have both conditions
    if 'high_interference' in epochs.event_id and len(epochs['high_interference']) > 0:
        evoked_high = epochs['high_interference'].average()
        evoked_high.apply_baseline((-0.2, 0))
        print(f"  High interference: {len(epochs['high_interference'])} trials")
    else:
        evoked_high = None
        print("  No high interference trials found")
    
    if 'low_interference' in epochs.event_id and len(epochs['low_interference']) > 0:
        evoked_low = epochs['low_interference'].average()
        evoked_low.apply_baseline((-0.2, 0))
        print(f"  Low interference: {len(epochs['low_interference'])} trials")
    else:
        evoked_low = None
        print("  No low interference trials found")
    
    return evoked_high, evoked_low

def compute_grand_average(all_high_evokeds, all_low_evokeds):
    """
    Compute grand averages across all subjects.
    """
    print("\nComputing grand averages...")
    
    if all_high_evokeds:
        print(f"  High interference: averaging {len(all_high_evokeds)} evokeds")
        grand_high = mne.grand_average(all_high_evokeds)
    else:
        print("  No high interference evokeds to average")
        grand_high = None
    
    if all_low_evokeds:
        print(f"  Low interference: averaging {len(all_low_evokeds)} evokeds")
        grand_low = mne.grand_average(all_low_evokeds)
    else:
        print("  No low interference evokeds to average")
        grand_low = None
    
    return grand_high, grand_low

def plot_grand_average(grand_high, grand_low):
    """
    Plot the grand average comparison.
    """
    print("\nGenerating grand average plot...")
    
    if grand_high is None and grand_low is None:
        print("No data to plot")
        return
    
    # Get available channels from the data
    if grand_high is not None:
        all_channels = grand_high.ch_names
    elif grand_low is not None:
        all_channels = grand_low.ch_names
    else:
        all_channels = []
    
    # Try to find central/parietal channels (standard 10-20 system)
    channels_to_plot = ['Cz', 'Pz', 'P3', 'P4', 'Fz', 'Fp1', 'Fp2']
    available_channels = [ch for ch in channels_to_plot if ch in all_channels]
    
    if not available_channels:
        # Use first 4 channels as fallback
        available_channels = all_channels[:4]
        print(f"Using first {len(available_channels)} available channels")
    else:
        print(f"Plotting channels: {available_channels}")
    
    fig, axes = plt.subplots(len(available_channels), 1, figsize=(10, 4 * len(available_channels)))
    if len(available_channels) == 1:
        axes = [axes]
    
    for idx, ch in enumerate(available_channels):
        ax = axes[idx]
        
        if grand_high is not None:
            ch_idx = grand_high.ch_names.index(ch)
            times = grand_high.times * 1000
            high_data_uv = grand_high.data[ch_idx, :] * 1e6
            ax.plot(times, high_data_uv, label='High Interference', color='red', linewidth=2)
        
        if grand_low is not None:
            ch_idx = grand_low.ch_names.index(ch)
            times = grand_low.times * 1000
            low_data_uv = grand_low.data[ch_idx, :] * 1e6
            ax.plot(times, low_data_uv, label='Low Interference', color='blue', linewidth=2)
        
        ax.axvspan(300, 500, alpha=0.2, color='yellow', label='N400 Window')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (uV)')
        ax.set_title(f'Grand Average ERP at {ch}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grand_average_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved grand average plot to: grand_average_comparison.png")
    plt.close()

def main():
    """
    Main grand average pipeline.
    """
    print("=" * 70)
    print("GRAND AVERAGE PIPELINE")
    print("=" * 70)
    
    # Step 1: Find EEG files
    eeg_files = find_eeg_files()
    
    if not eeg_files:
        print("No EEG files found. Exiting.")
        return
    
    # Step 2: Load tagged events if available
    tagged_events = load_tagged_events()
    
    # Step 3: Initialize lists
    all_high_evokeds = []
    all_low_evokeds = []
    
    # Step 4: Process each file
    skipped_files = 0
    for eeg_file in eeg_files:
        evoked_high, evoked_low = process_single_subject(eeg_file, tagged_events)
        
        if evoked_high is None and evoked_low is None:
            skipped_files += 1
            continue
        
        if evoked_high is not None:
            all_high_evokeds.append(evoked_high)
        
        if evoked_low is not None:
            all_low_evokeds.append(evoked_low)
    
    # Step 5: Compute grand averages
    grand_high, grand_low = compute_grand_average(all_high_evokeds, all_low_evokeds)
    
    # Step 6: Plot results
    plot_grand_average(grand_high, grand_low)
    
    # Summary
    print("\n" + "=" * 70)
    print("GRAND AVERAGE COMPLETE")
    print("=" * 70)
    print(f"\nTotal files found: {len(eeg_files)}")
    print(f"Files successfully processed: {len(eeg_files) - skipped_files}")
    print(f"Files skipped (corrupted): {skipped_files}")
    print(f"High interference evokeds: {len(all_high_evokeds)}")
    print(f"Low interference evokeds: {len(all_low_evokeds)}")
    print("\nOutput file:")
    print("  - grand_average_comparison.png")

if __name__ == "__main__":
    main()
