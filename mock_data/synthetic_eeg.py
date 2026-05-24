"""
Create Synthetic EEG Data
Generates mock EEG data that matches our word recognition events timeline.
"""

import logging
import mne
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

def create_synthetic_eeg() -> mne.io.Raw:
    """Create synthetic EEG data matching our word recognition events."""
    logging.basicConfig(level=logging.INFO)
    
    events_file = Path("data/sub-01/eeg/sub-01_task-wordrecognition_events.tsv")
    events_df = pd.read_csv(events_file, sep='\t')
    
    total_duration = (events_df['onset'].max() + events_df['duration'].max() + 5000) / 1000
    
    logger.info(f"Total duration: {total_duration:.1f} seconds")
    logger.info(f"Number of events: {len(events_df)}")
    
    sfreq = 500
    n_channels = 19
    n_times = int(total_duration * sfreq)
    
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4',
                'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz']
    ch_types = ['eeg'] * n_channels
    
    t = np.arange(n_times) / sfreq
    data = np.random.randn(n_channels, n_times) * 1e-6
    
    for i in range(1, 50):
        freq = i * 0.5
        amplitude = 2e-6 / freq
        data += amplitude * np.sin(2 * np.pi * freq * t) * np.random.randn(n_channels, n_times)
    
    for idx, row in events_df.iterrows():
        onset_sample = int(row['onset'] / 1000 * sfreq)
        duration_samples = int(row['duration'] / 1000 * sfreq)
        
        erp_window = np.arange(-100, 500)
        erp_time = erp_window / sfreq
        n400 = -5e-6 * np.exp(-((erp_time - 0.4) ** 2) / (2 * 0.1 ** 2))
        p200 = 3e-6 * np.exp(-((erp_time - 0.2) ** 2) / (2 * 0.08 ** 2))
        erp = n400 + p200
        
        for ch_idx, ch_name in enumerate(ch_names):
            if ch_name in ['Cz', 'Pz', 'P3', 'P4', 'C3', 'C4']:
                data[ch_idx, onset_sample:onset_sample+len(erp)] += erp * 2
            else:
                data[ch_idx, onset_sample:onset_sample+len(erp)] += erp * 0.5
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    output_file = Path("data/sub-01/eeg/sub-01_task-wordrecognition_eeg.fif")
    raw.save(output_file, overwrite=True)
    logger.info(f"Saved synthetic EEG to: {output_file}")
    logger.info(f"Shape: {data.shape} (channels x timepoints)")
    logger.info(f"Duration: {raw.times[-1]:.1f} seconds")
    logger.info(f"Sampling frequency: {sfreq} Hz")
    
    return raw

if __name__ == "__main__":
    create_synthetic_eeg()
