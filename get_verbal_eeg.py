"""
Download ERP-CORE EEG Dataset from OpenNeuro
Downloads the ERP-CORE dataset (ds000247) for all 40 subjects in BIDS format.
"""

import logging
import openneuro
from openneuro import download
from pathlib import Path
import shutil
import pandas as pd

logger = logging.getLogger(__name__)


def download_single_subject_eeg(
    dataset_id: str = "ds000247",
    subject_id: str = "01",
    target_dir: str = "./data",
    tag: str = "2.0.0",
    task_filter: str | None = None
) -> str | None:
    """Download a single subject's EEG data from OpenNeuro in BIDS format."""
    
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    if task_filter:
        include_pattern = f"sub-{subject_id}/**/*{task_filter}**"
    else:
        include_pattern = f"sub-{subject_id}/**"
    
    logger.info(f"Dataset ID: {dataset_id}")
    logger.info(f"Subject: sub-{subject_id}")
    logger.info(f"Target directory: {target_dir}")
    logger.info(f"Include pattern: {include_pattern}")
    
    logger.info("Connecting to OpenNeuro...")
    logger.info("Downloading subject data (this may take several minutes)...")
    
    try:
        download(
            dataset=dataset_id,
            target_dir=target_dir,
            include=include_pattern,
            tag=tag
        )
        
        logger.info("Download completed successfully!")
        
        downloaded_path = target_path / dataset_id
        subject_target = target_path / f"sub-{subject_id}"
        
        if (target_path / f"sub-{subject_id}").exists():
            logger.info(f"Data already organized at: {subject_target.absolute()}")
            list_files_recursive(subject_target)
            check_essential_files(subject_target)
            return str(subject_target)
        
        elif downloaded_path.exists():
            subject_source = downloaded_path / f"sub-{subject_id}"
            
            if subject_source.exists():
                if subject_target.exists():
                    shutil.rmtree(subject_target)
                shutil.move(str(subject_source), str(subject_target))
                
                for file in ['dataset_description.json', 'participants.tsv', 'participants.json', 'README', 'CHANGES']:
                    src = downloaded_path / file
                    if src.exists():
                        shutil.move(str(src), str(target_path / file))
                
                try:
                    remaining_files = list(downloaded_path.iterdir())
                    if len(remaining_files) == 0:
                        downloaded_path.rmdir()
                    elif len(remaining_files) == 1 and remaining_files[0].name == '.git':
                        shutil.rmtree(downloaded_path)
                except OSError:
                    pass
                
                logger.info(f"Data organized at: {subject_target.absolute()}")
                list_files_recursive(subject_target)
                check_essential_files(subject_target)
                return str(subject_target)
            else:
                logger.warning(f"Subject folder sub-{subject_id} not found in download")
                list_files_recursive(downloaded_path)
                return str(downloaded_path)
        else:
            logger.warning("Download completed but directory not found")
            return None
            
    except Exception as e:
        logger.error(f"Error during download: {e}")
        logger.warning("Troubleshooting tips:")
        logger.warning("1. Check your internet connection")
        logger.warning("2. Verify the dataset ID is correct")
        logger.warning("3. Ensure you have write permissions for the target directory")
        logger.warning("4. The dataset might not exist or be publicly accessible")
        return None


def list_files_recursive(path: Path, max_depth: int = 4, current_depth: int = 0) -> None:
    """Recursively list files in a directory with indentation."""
    if current_depth > max_depth:
        return
    
    try:
        for item in sorted(path.iterdir()):
            indent = "  " * current_depth
            if item.is_dir():
                logger.info(f"{indent}{item.name}/")
                list_files_recursive(item, max_depth, current_depth + 1)
            else:
                logger.info(f"{indent}{item.name}")
    except PermissionError:
        logger.warning(f"{indent}[Permission denied]")


def check_essential_files(subject_path: Path) -> None:
    """Check if essential EEG files are present."""
    logger.info("Checking for essential files:")
    
    eeg_extensions = ['.fif', '.edf', '.bdf', '.vhdr', '.set', '.eeg']
    eeg_files = []
    events_files = []
    
    for file in subject_path.rglob("*"):
        if file.is_file():
            if file.suffix.lower() in eeg_extensions:
                eeg_files.append(file)
            if file.name.endswith('_events.tsv'):
                events_files.append(file)
    
    logger.info(f"Raw EEG data files found: {len(eeg_files)}")
    for f in eeg_files:
        logger.info(f"  - {f.relative_to(subject_path)}")
    
    logger.info(f"Events TSV files found: {len(events_files)}")
    for f in events_files:
        logger.info(f"  - {f.relative_to(subject_path)}")
    
    if eeg_files:
        logger.info("Raw EEG data files present")
    else:
        logger.warning("No raw EEG data files found")
    
    if events_files:
        logger.info("Events TSV files present")
    else:
        logger.warning("No events TSV files found")


def try_alternative_datasets(subject_id: str = "01", target_dir: str = "./data") -> str | None:
    """Try alternative verbal memory/word recognition datasets if primary fails."""
    logger.info("Trying alternative datasets...")
    
    alternatives = [
        ("ds003620", "1.1.1", "Language production and comprehension (oddball)"),
        ("ds002410", "1.0.0", "EEG resting state"),
        ("ds001465", "1.0.0", "EEG dataset"),
    ]
    
    for dataset_id, tag, description in alternatives:
        logger.info(f"Trying {dataset_id} (tag: {tag}): {description}")
        result = download_single_subject_eeg(
            dataset_id=dataset_id,
            subject_id=subject_id,
            target_dir=target_dir,
            tag=tag,
            task_filter=None  # Download all files first
        )
        if result:
            logger.info(f"Successfully downloaded from {dataset_id}")
            return result
    
    logger.warning("All alternative datasets failed")
    return None


def main() -> None:
    """Download ERP-CORE dataset (ds000247) for all 40 subjects."""
    logging.basicConfig(level=logging.INFO)
    
    dataset_id = "ds000247"
    tag = "1.0.2"
    target_dir = "./data"
    total_subjects = 40
    
    logger.info(f"Downloading ERP-CORE dataset {dataset_id} (tag: {tag})")
    logger.info(f"Target: {total_subjects} subjects")
    
    data_dir = Path(target_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    successful_downloads = 0
    skipped_subjects = 0
    failed_downloads = 0
    
    for i in range(1, total_subjects + 1):
        subject_id = f"{i:02d}"
        subject_path = data_dir / f"sub-{subject_id}"
        
        if subject_path.exists():
            logger.info(f"Skipping sub-{subject_id} (already exists)")
            skipped_subjects += 1
            continue
        
        logger.info(f"Downloading sub-{subject_id} ({i}/{total_subjects})...")
        
        result = download_single_subject_eeg(
            dataset_id=dataset_id,
            subject_id=subject_id,
            target_dir=target_dir,
            tag=tag,
            task_filter=None
        )
        
        if result:
            successful_downloads += 1
            logger.info(f"Successfully downloaded sub-{subject_id}")
        else:
            failed_downloads += 1
            logger.warning(f"Failed to download sub-{subject_id}")
    
    logger.info("=" * 70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total subjects: {total_subjects}")
    logger.info(f"Successfully downloaded: {successful_downloads}")
    logger.info(f"Skipped (already exists): {skipped_subjects}")
    logger.info(f"Failed: {failed_downloads}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
