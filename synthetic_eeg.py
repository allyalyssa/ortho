"""
Create Synthetic EEG Data
Generates mock EEG data that matches our word recognition events timeline.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path

def create_synthetic_eeg():
    """
    Create synthetic EEG data matching our word recognition events.
    """
    print("Creating synthetic EEG data...")
    
    # Load events to get timing
    events_file = Path("data/sub-01/eeg/sub-01_task-wordrecognition_events.tsv")
    events_df = pd.read_csv(events_file, sep='\t')
    
    # Calculate total duration (last onset + duration + buffer)
    total_duration = (events_df['onset'].max() + events_df['duration'].max() + 5000) / 1000  # Convert to seconds
    
    print(f"Total duration: {total_duration:.1f} seconds")
    print(f"Number of events: {len(events_df)}")
    
    # EEG parameters
    sfreq = 500  # Sampling frequency (Hz)
    n_channels = 19  # Standard 10-20 system
    n_times = int(total_duration * sfreq)
    
    # Channel names (10-20 system)
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4',
                'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz']
    
    # Channel types
    ch_types = ['eeg'] * n_channels
    
    # Create synthetic data
    # Baseline: random noise + 1/f pink noise
    t = np.arange(n_times) / sfreq
    data = np.random.randn(n_channels, n_times) * 1e-6  # Random noise in Volts
    
    # Add 1/f pink noise (more power at low frequencies)
    for i in range(1, 50):
        freq = i * 0.5
        amplitude = 2e-6 / freq  # Amplitude in Volts
        data += amplitude * np.sin(2 * np.pi * freq * t) * np.random.randn(n_channels, n_times)
    
    # Add simulated ERP responses for each word event
    for idx, row in events_df.iterrows():
        onset_sample = int(row['onset'] / 1000 * sfreq)
        duration_samples = int(row['duration'] / 1000 * sfreq)
        
        # Create a simulated N400-like response (negative peak around 400ms)
        erp_window = np.arange(-100, 500)  # -200ms to +800ms relative to onset
        erp_time = erp_window / sfreq
        
        # N400: negative peak at 400ms (in Volts)
        n400 = -5e-6 * np.exp(-((erp_time - 0.4) ** 2) / (2 * 0.1 ** 2))
        
        # P200: positive peak at 200ms (in Volts)
        p200 = 3e-6 * np.exp(-((erp_time - 0.2) ** 2) / (2 * 0.08 ** 2))
        
        # Combine ERP components
        erp = n400 + p200
        
        # Add to data (stronger at central/parietal channels)
        for ch_idx, ch_name in enumerate(ch_names):
            if ch_name in ['Cz', 'Pz', 'P3', 'P4', 'C3', 'C4']:
                # Central/parietal channels get stronger ERPs
                data[ch_idx, onset_sample:onset_sample+len(erp)] += erp * 2
            else:
                data[ch_idx, onset_sample:onset_sample+len(erp)] += erp * 0.5
    
    # Create MNE Raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    
    # Add standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # Save as FIF file
    output_file = Path("data/sub-01/eeg/sub-01_task-wordrecognition_eeg.fif")
    raw.save(output_file, overwrite=True)
    
    print(f"Saved synthetic EEG to: {output_file}")
    print(f"Shape: {data.shape} (channels x timepoints)")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    print(f"Sampling frequency: {sfreq} Hz")
    
    return raw

if __name__ == "__main__":
    create_synthetic_eeg()
