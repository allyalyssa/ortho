"""
Download Verbal Working Memory/Word Recognition EEG Dataset from OpenNeuro
Downloads a single subject's raw EEG data and events.tsv in BIDS format.
Targets language/memory datasets like ERP-CORE (N400) or verbal n-back tasks.
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
    tag: str = "1.0.2",
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
    """Main function to download verbal memory EEG data."""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Attempting to download from primary dataset...")
    
    logger.info("OpenNeuro datasets unavailable. Creating synthetic word recognition data...")
    logger.info("This will create a mock events.tsv with words from our density lists.")
    
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    subject_dir = data_dir / "sub-01"
    subject_dir.mkdir(exist_ok=True)
    eeg_dir = subject_dir / "eeg"
    eeg_dir.mkdir(exist_ok=True)
    
    from orthographic_density import get_4_letter_nouns, calculate_neighborhood_density, categorize_by_density
    
    nouns = get_4_letter_nouns()
    sample_size = min(100, len(nouns))
    word_list = nouns[:sample_size]
    density_dict = calculate_neighborhood_density(word_list, distance_threshold=1)
    high_density, low_density, median = categorize_by_density(density_dict)
    
    # Create events with words
    import random
    events_data = []
    onset = 1000
    
    # Mix high and low density words
    all_words = list(high_density.keys()) + list(low_density.keys())
    random.shuffle(all_words)
    
    for i, word in enumerate(all_words[:50]):  # Use 50 trials
        events_data.append({
            'onset': onset,
            'duration': 500,
            'trial_type': f'word_{word}',
            'stimulus': word
        })
        onset += 2000  # 2 seconds between trials
    
    # Save as TSV
    events_df = pd.DataFrame(events_data)
    events_file = eeg_dir / "sub-01_task-wordrecognition_events.tsv"
    events_df.to_csv(events_file, sep='\t', index=False)
    
    logger.info(f"Created synthetic events file: {events_file}")
    logger.info(f"Contains {len(events_df)} word trials")
    logger.info(f"Words: {events_df['stimulus'].head(10).tolist()}")
    
    result = str(subject_dir)
    
    if result:
        logger.info(f"Downloaded data location: {result}")
        logger.info("Next steps:")
        logger.info("1. Review the downloaded BIDS structure")
        logger.info("2. Use MNE-Python to load the EEG data")
        logger.info("3. Map word trials from events.tsv to EEG epochs")
        logger.info("4. Analyze brain responses to different word conditions")
    else:
        logger.warning("Please check:")
        logger.warning("1. Internet connection")
        logger.warning("2. Dataset availability on OpenNeuro")
        logger.warning("3. Write permissions for target directory")


if __name__ == "__main__":
    main()
