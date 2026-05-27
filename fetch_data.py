"""
Download Real EEG Data from OpenNeuro
Downloads actual EEG data from OpenNeuro dataset ds004306 using openneuro-py.
"""

import logging
import openneuro
from openneuro import download
from pathlib import Path

logger = logging.getLogger(__name__)

def download_openneuro_dataset() -> None:
    """Download EEG data from OpenNeuro dataset ds004306 for multiple subjects."""
    logger.info("Downloading real EEG data from OpenNeuro dataset ds004306...")
    
    dataset_id = "ds004306"
    tag = "1.0.0"
    subjects = ["01", "02", "03"]
    target_dir = Path("./data")
    target_dir.mkdir(exist_ok=True)
    downloaded_count = 0
    
    for subject_id in subjects:
        logger.info(f"Downloading sub-{subject_id}...")
        
        try:
            download(
                dataset=dataset_id,
                target_dir=str(target_dir),
                tag=tag,
                include=f"sub-{subject_id}/**"
            )
            logger.info(f"Successfully downloaded sub-{subject_id}")
            downloaded_count += 1
        except Exception as e:
            logger.error(f"Error downloading sub-{subject_id}: {e}")
            continue
    
    logger.info(f"Subjects attempted: {len(subjects)}")
    logger.info(f"Subjects downloaded: {downloaded_count}")
    
    data_dir = Path("./data")
    if data_dir.exists():
        sub_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        logger.info(f"Subject directories found: {len(sub_dirs)}")
        for sub_dir in sub_dirs:
            logger.info(f"  - {sub_dir.name}")
            eeg_files = list(sub_dir.rglob("*.fif")) + list(sub_dir.rglob("*.edf"))
            if eeg_files:
                logger.info(f"    EEG files: {len(eeg_files)}")
    
    if downloaded_count > 0:
        logger.info("Download complete!")
    else:
        logger.warning("Download failed. Dataset may not be accessible.")

if __name__ == "__main__":
    download_openneuro_dataset()
