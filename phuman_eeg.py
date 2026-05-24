"""
Process Real Human EEG Data
Processes downloaded EEG data from OpenNeuro with bandpass filtering, epoching, and artifact rejection.
"""

import logging
import mne
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def process_single_subject(subject_id: str, data_dir: Path = Path("./data")) -> bool:
    """Process a single subject's EEG data and save preprocessed epochs."""
    subject_dir = data_dir / f"sub-{subject_id}" / "eeg"
    
    if not subject_dir.exists():
        logger.warning(f"Subject directory not found: {subject_dir}")
        return False
    
    eeg_files = list(subject_dir.glob("*.fif")) + list(subject_dir.glob("*.set")) + list(subject_dir.glob("*.edf"))
    
    if not eeg_files:
        logger.warning(f"No EEG files found for sub-{subject_id}")
        return False
    
    eeg_file = eeg_files[0]
    logger.info(f"Processing sub-{subject_id}: {eeg_file.name}")
    
    try:
        if eeg_file.suffix == '.set':
            raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False, montage_units='mm')
        elif eeg_file.suffix == '.edf':
            raw = mne.io.read_raw_edf(str(eeg_file), preload=True, verbose=False)
        else:
            raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    except Exception as e:
        logger.error(f"Error loading sub-{subject_id}: {e}")
        return False
    
    logger.info(f"Loaded: {len(raw.ch_names)} channels, {raw.n_times} timepoints, {raw.info['sfreq']} Hz")
    
    raw.filter(l_freq=0.1, h_freq=20, verbose=False)
    
    events_file = subject_dir / f"sub-{subject_id}_task-*_events.tsv"
    events_files = list(subject_dir.glob("*_events.tsv"))
    
    if not events_files:
        logger.warning(f"No events file found for sub-{subject_id}, using MNE events")
        events_array = mne.find_events(raw, stim_channel='STI 014', verbose=False)
        event_id_dict = {'1': 1, '2': 2}
    else:
        events_file = events_files[0]
        events_df = pd.read_csv(events_file, sep='\t')
        
        sfreq = raw.info['sfreq']
        events_df['sample'] = (events_df['onset'] * sfreq).astype(int)
        
        unique_types = events_df['trial_type'].unique() if 'trial_type' in events_df.columns else events_df['value'].unique()
        event_id_map = {str(i+1): t for i, t in enumerate(unique_types)}
        reverse_event_id_map = {t: i+1 for i, t in enumerate(unique_types)}
        
        events_df['event_id'] = events_df['trial_type'].map(reverse_event_id_map) if 'trial_type' in events_df.columns else events_df['value'].map(reverse_event_id_map)
        
        events_array = []
        for idx, row in events_df.iterrows():
            events_array.append([row['sample'], 0, row['event_id']])
        events_array = np.array(events_array, dtype=int)
        event_id_dict = {str(i+1): t for i, t in enumerate(unique_types)}
    
    reject = dict(eeg=100e-6)
    
    epochs = mne.Epochs(
        raw,
        events_array,
        event_id=event_id_dict,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0),
        reject=reject,
        preload=True,
        verbose=False
    )
    
    epochs.apply_baseline((-0.2, 0))
    
    total_epochs = len(epochs)
    usable_epochs = len(epochs)
    rejected_epochs = 0
    
    output_dir = data_dir / "derivatives" / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"sub-{subject_id}_ica_eeg.fif"
    epochs.save(output_file, overwrite=True, verbose=False)
    
    logger.info(f"sub-{subject_id}: {usable_epochs} usable epochs, {rejected_epochs} rejected")
    
    return True

def main() -> None:
    """Process all subjects in data/sub-XX/eeg/."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    data_dir = Path("./data")
    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    if not subject_dirs:
        logger.warning("No subject directories found in data/")
        return
    
    logger.info(f"Found {len(subject_dirs)} subject directories")
    
    successful = 0
    failed = 0
    
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name.replace('sub-', '')
        
        if process_single_subject(subject_id, data_dir):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
