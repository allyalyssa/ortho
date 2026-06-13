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
    subject_dir = data_dir / "erpcore" / "N400" / f"sub-{subject_id}" / "eeg"
    
    if not subject_dir.exists():
        logger.warning(f"Subject directory not found: {subject_dir}")
        return False
    
    eeg_files = list(subject_dir.glob("*_task-N400_eeg.set"))
    
    if not eeg_files:
        logger.warning(f"No EEG files found for sub-{subject_id}")
        return False
    
    eeg_file = eeg_files[0]
    logger.info(f"Processing sub-{subject_id}: {eeg_file.name}")
    
    try:
        raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False, montage_units='mm')
    except (IOError, ValueError) as e:
        logger.error(f"Error loading sub-{subject_id}: {e}")
        return False
    
    logger.info(f"Loaded: {len(raw.ch_names)} channels, {raw.n_times} timepoints, {raw.info['sfreq']} Hz")
    
    raw.filter(l_freq=0.1, h_freq=20, verbose=False)
    
    # ICA for artifact removal
    logger.info(f"Fitting ICA with n_components=15")
    ica = mne.preprocessing.ICA(n_components=15, random_state=42, max_iter=800)
    ica.fit(raw, verbose=False)
    
    # Detect EOG channels
    eog_channels = [ch for ch in raw.ch_names if 'EOG' in ch.upper() or 'HEOG' in ch.upper() or 'VEOG' in ch.upper()]
    
    if eog_channels:
        logger.info(f"Found EOG channels: {eog_channels}")
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels[0], verbose=False)
        logger.info(f"Detected {len(eog_indices)} EOG-related components")
    else:
        logger.warning("No EOG channels found, using correlation-based detection")
        # Use correlation with frontal channels as fallback
        frontal_channels = [ch for ch in raw.ch_names if 'Fp' in ch or 'Fz' in ch]
        if frontal_channels:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=frontal_channels[0], verbose=False)
            logger.info(f"Detected {len(eog_indices)} frontal-correlated components")
        else:
            eog_indices = []
            logger.warning("No frontal channels found, skipping EOG detection")
    
    if eog_indices:
        ica.exclude = eog_indices
        raw = ica.apply(raw, verbose=False)
        logger.info(f"Removed {len(eog_indices)} ICA components")
    else:
        logger.info("No ICA components removed")
    
    # Get events from annotations (ERP CORE data has proper event codes)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # Filter for N400 target codes only
    RELATED_CODES = {'211', '212'}
    UNRELATED_CODES = {'221', '222'}
    TARGET_CODES = RELATED_CODES | UNRELATED_CODES
    
    target_id = {k: v for k, v in event_id.items() if str(k) in TARGET_CODES}
    
    if not target_id:
        logger.warning(f"No target events found for sub-{subject_id}")
        return False
    
    events_array = events
    event_id_dict = target_id
    
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
    
    usable_epochs_after_ica = len(epochs)
    rejected_epochs_after_ica = len(epochs.drop_log) - usable_epochs_after_ica
    
    output_dir = data_dir / "derivatives" / "preprocessed_ica"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"sub-{subject_id}_N400_ica-epo.fif"
    epochs.save(output_file, overwrite=True, verbose=False)
    
    logger.info(f"sub-{subject_id}: {usable_epochs_after_ica} usable epochs after ICA, {rejected_epochs_after_ica} rejected")
    
    return True

def main() -> None:
    """Process all subjects in data/erpcore/N400/sub-XXX/eeg/."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    data_dir = Path("./data")
    erpcore_dir = data_dir / "erpcore" / "N400"
    subject_dirs = sorted([d for d in erpcore_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    
    if not subject_dirs:
        logger.warning("No subject directories found in data/erpcore/N400/")
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


if __name__ == "__main__":
    main()
