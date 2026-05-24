"""
Download MNE-Python EEG Dataset
Downloads an EEG dataset from MNE-Python and organizes it into the data directory.
"""

import logging
import mne
from mne.datasets import sample
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

def download_sample_eeg() -> str | None:
    """Download the MNE sample dataset and organize it."""
    logger.info("Downloading MNE sample dataset...")
    
    data_path = sample.data_path()
    logger.info(f"Dataset downloaded to: {data_path}")
    
    target_dir = Path("./data")
    target_dir.mkdir(exist_ok=True)
    
    sample_dir = Path(data_path) / "MEG" / "sample"
    
    raw_file = sample_dir / "sample_audvis_raw.fif"
    
    if raw_file.exists():
        logger.info(f"Found raw data file: {raw_file.name}")
        
        # Load the raw data and extract EEG channels
        raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
        
        raw_eeg = raw.pick_types(meg=False, eeg=True, eog=False, stim=False)
        
        sub_dir = target_dir / "sub-01"
        sub_dir.mkdir(exist_ok=True)
        eeg_dir = sub_dir / "eeg"
        eeg_dir.mkdir(exist_ok=True)
        
        output_file = eeg_dir / "sub-01_task-visual_eeg.fif"
        raw_eeg.save(output_file, overwrite=True)
        logger.info(f"Saved EEG data to: {output_file.name}")
        
        events_file = sample_dir / "sample_audvis_raw-eve.fif"
        if events_file.exists():
            dest = eeg_dir / "sub-01_task-visual_events.fif"
            shutil.copy(events_file, dest)
            logger.info("Copied events file")
        
        logger.info(f"Data organized in: {eeg_dir}")
        logger.info("Download complete!")
        return str(eeg_dir)
    else:
        logger.warning("No raw data file found")
        return None

if __name__ == "__main__":
    download_sample_eeg()
