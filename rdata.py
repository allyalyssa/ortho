"""
Download Real EEG Data from OpenNeuro
Downloads actual EEG data from OpenNeuro dataset ds004306 using openneuro-py.
"""

import openneuro
from openneuro import download
from pathlib import Path

def download_openneuro_dataset():
    """
    Download EEG data from OpenNeuro dataset ds004306 for multiple subjects.
    """
    print("Downloading real EEG data from OpenNeuro dataset ds004306...")
    print("Using openneuro-py library")
    
    # Dataset information
    dataset_id = "ds004306"
    tag = "1.0.0"
    subjects = ["01", "02", "03"]
    
    # Create data directory
    target_dir = Path("./data")
    target_dir.mkdir(exist_ok=True)
    
    # Download each subject
    downloaded_count = 0
    
    for subject_id in subjects:
        print(f"\nDownloading sub-{subject_id}...")
        
        try:
            download(
                dataset=dataset_id,
                target_dir=str(target_dir),
                tag=tag,
                include=f"sub-{subject_id}/**"
            )
            print(f"Successfully downloaded sub-{subject_id}")
            downloaded_count += 1
        except Exception as e:
            print(f"Error downloading sub-{subject_id}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Subjects attempted: {len(subjects)}")
    print(f"Subjects downloaded: {downloaded_count}")
    
    # Check what was downloaded
    data_dir = Path("./data")
    if data_dir.exists():
        sub_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')]
        print(f"\nSubject directories found: {len(sub_dirs)}")
        for sub_dir in sub_dirs:
            print(f"  - {sub_dir.name}")
            eeg_files = list(sub_dir.rglob("*.fif")) + list(sub_dir.rglob("*.edf"))
            if eeg_files:
                print(f"    EEG files: {len(eeg_files)}")
    
    if downloaded_count > 0:
        print("\nDownload complete!")
    else:
        print("\nDownload failed. Dataset may not be accessible.")

if __name__ == "__main__":
    download_openneuro_dataset()
