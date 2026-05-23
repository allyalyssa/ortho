"""
Download MNE-Python EEG Dataset
Downloads an EEG dataset from MNE-Python and organizes it into the data directory.
"""

import mne
from mne.datasets import sample
from pathlib import Path
import shutil

def download_sample_eeg():
    """
    Download the MNE sample dataset which contains EEG data and organize it.
    """
    print("Downloading MNE sample dataset...")
    print("This dataset contains EEG and MEG data from a visual stimulation task.")
    
    # Download the sample dataset
    data_path = sample.data_path()
    print(f"\nDataset downloaded to: {data_path}")
    
    # Create our data directory structure
    target_dir = Path("./data")
    target_dir.mkdir(exist_ok=True)
    
    # The sample dataset has EEG data in the MEG directory
    sample_dir = Path(data_path) / "MEG" / "sample"
    
    # Load the raw file which contains EEG data
    raw_file = sample_dir / "sample_audvis_raw.fif"
    
    if raw_file.exists():
        print(f"\nFound raw data file: {raw_file.name}")
        
        # Load the raw data and extract EEG channels
        raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
        
        # Pick only EEG channels
        raw_eeg = raw.pick_types(meg=False, eeg=True, eog=False, stim=False)
        
        # Create sub-01 directory structure
        sub_dir = target_dir / "sub-01"
        sub_dir.mkdir(exist_ok=True)
        eeg_dir = sub_dir / "eeg"
        eeg_dir.mkdir(exist_ok=True)
        
        # Save the EEG data
        output_file = eeg_dir / "sub-01_task-visual_eeg.fif"
        raw_eeg.save(output_file, overwrite=True)
        print(f"Saved EEG data to: {output_file.name}")
        
        # Also save the events
        events_file = sample_dir / "sample_audvis_raw-eve.fif"
        if events_file.exists():
            dest = eeg_dir / "sub-01_task-visual_events.fif"
            shutil.copy(events_file, dest)
            print(f"Copied events file")
        
        print(f"\nData organized in: {eeg_dir}")
        print("\nDownload complete!")
        return str(eeg_dir)
    else:
        print("\nNo raw data file found")
        return None

if __name__ == "__main__":
    download_sample_eeg()
