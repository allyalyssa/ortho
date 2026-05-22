# EEG Data Download Summary

## Successfully Downloaded Dataset

**Dataset ID:** ds003620  
**Name:** Runabout: A mobile EEG study of auditory oddball processing in laboratory and real-world conditions  
**Version:** 1.1.1  
**Subject:** sub-01  
**Task:** oddball (auditory oddball paradigm)

## Downloaded Files Structure

```
data/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── README
├── CHANGES
└── sub-01/
    └── eeg/
        ├── sub-01_task-oddball_channels.tsv
        ├── sub-01_task-oddball_eeg.eeg
        ├── sub-01_task-oddball_eeg.json
        ├── sub-01_task-oddball_eeg.vhdr
        ├── sub-01_task-oddball_eeg.vmrk
        ├── sub-01_task-oddball_electrodes.json
        ├── sub-01_task-oddball_electrodes.tsv
        └── sub-01_task-oddball_events.tsv
```

## Key Files

### Raw EEG Data
- **sub-01_task-oddball_eeg.vhdr** - BrainVision header file (use this to load the data)
- **sub-01_task-oddball_eeg.eeg** - Binary EEG data file
- **sub-01_task-oddball_eeg.vmrk** - BrainVision marker file

### Events/Timeline
- **sub-01_task-oddball_events.tsv** - Event markers with onset, duration, and trial_type columns

### Metadata
- **sub-01_task-oddball_channels.tsv** - Channel information
- **sub-01_task-oddball_electrodes.tsv** - Electrode locations
- **sub-01_task-oddball_eeg.json** - BIDS JSON metadata

## Important Notes

### Task Type
This dataset contains an **auditory oddball task**, not a word recognition or verbal memory task. The events contain:
- **"S  1"** - Standard stimuli (frequent)
- **empty** - Empty/placeholder events

While this isn't the ideal verbal memory/word recognition task you requested, it provides:
- Complete BIDS-formatted EEG data
- Events file for mapping trials to brain waves
- Same structure you would use for word recognition tasks

### How to Load and Analyze

```python
import mne

# Load the raw EEG data
raw = mne.io.read_raw_brainvision(
    'data/sub-01/eeg/sub-01_task-oddball_eeg.vhdr',
    preload=True
)

# Load events
events, event_id = mne.events_from_annotations(raw)

# Or load from TSV file
import pandas as pd
events_df = pd.read_csv('data/sub-01/eeg/sub-01_task-oddball_events.tsv', sep='\t')

# Epoch around events
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8)

# Analyze ERPs
evoked = epochs['S  1'].average()
evoked.plot()
```

## Finding Word Recognition/Verbal Memory Datasets

To find datasets more suitable for your psycholinguistics project:

1. **Search OpenNeuro directly:**
   - Visit https://openneuro.org
   - Search for: "N400", "semantic", "word recognition", "lexical decision", "verbal memory"
   - Filter by: EEG modality

2. **Known datasets to try:**
   - **ERP-CORE (ds000247)** - Contains N400 semantic processing task
   - **Various language production datasets** - Search for "language" + "EEG"

3. **Modify the download script:**
   - Change the `dataset_id` in `download_verbal_eeg.py`
   - Use specific version tags (check OpenNeuro for available versions)
   - Filter by specific tasks if the dataset has multiple tasks

## Next Steps

1. **Explore the current data** to understand BIDS structure
2. **Practice event-to-EEG mapping** using this oddball dataset
3. **Search OpenNeuro** for a more suitable verbal memory/word recognition dataset
4. **Modify the script** to download the specific dataset you find

## Script Location

The download script is at: `download_verbal_eeg.py`

To download a different dataset, modify the `dataset_id` and `tag` parameters in the `main()` function.
