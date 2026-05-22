# Orthographic Neighborhood Density & EEG Data Analysis

Python scripts for psycholinguistics research including orthographic neighborhood density analysis and EEG dataset downloading from OpenNeuro.

## Overview

This script:
1. Extracts 4-letter English nouns from WordNet
2. Calculates Levenshtein distance between all word pairs
3. Computes orthographic neighborhood density (number of neighbors at distance ≤ 1)
4. Categorizes words into 'high density' and 'low density' groups based on median split
5. Saves results to `neighborhood_density_results.txt`

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Scripts

### 1. Orthographic Density Analysis (`orthographic_density.py`)

Analyzes orthographic neighborhood density of 4-letter English nouns.

**Usage:**
```bash
python orthographic_density.py
```

**What it does:**
1. Extracts 4-letter English nouns from WordNet
2. Calculates Levenshtein distance between all word pairs
3. Computes orthographic neighborhood density (number of neighbors at distance ≤ 1)
4. Categorizes words into 'high density' and 'low density' groups based on median split
5. Saves results to `neighborhood_density_results.txt`

### 2. EEG Data Download (`download_verbal_eeg.py`)

Downloads EEG datasets from OpenNeuro in BIDS format for psycholinguistics research.

**Usage:**
```bash
python download_verbal_eeg.py
```

**What it does:**
- Downloads EEG data from OpenNeuro using openneuro-py
- Downloads single-subject data in BIDS (Brain Imaging Data Structure) format
- Includes raw EEG files (.vhdr, .eeg, .vmkr) and events.tsv for trial mapping
- Organizes data in `data/sub-01/` structure

**Currently Downloaded:**
- **Dataset:** ds003620 (Runabout: Auditory oddball processing)
- **Task:** Oddball paradigm (not word recognition, but useful for testing pipeline)
- **Files:** Raw EEG data, events.tsv, channel info, electrode locations
- **Location:** `data/sub-01/eeg/`

**Note:** The current dataset is an auditory oddball task. For verbal memory/word recognition tasks, search OpenNeuro for datasets like:
- ERP-CORE (ds000247) - N400 semantic processing
- Language production datasets
- Lexical decision tasks

**Custom usage:**
```python
from download_verbal_eeg import download_single_subject_eeg

# Download specific dataset
download_single_subject_eeg(
    dataset_id="ds000247",  # Change to desired dataset
    subject_id="01",
    target_dir="./data",
    tag="1.0.2"
)
```

See `DOWNLOAD_SUMMARY.md` for details on the downloaded data and how to find verbal memory datasets.

## Output

### Orthographic Density Script
- **Console**: Summary statistics and categorized word lists
- **File**: `neighborhood_density_results.txt` with complete results including:
  - Total words analyzed
  - Median neighborhood density
  - High density group (words with neighbors ≥ median)
  - Low density group (words with neighbors < median)

### EEG Download Script
- **Console**: Download progress and file structure overview
- **Directory**: BIDS-formatted EEG data in `./eeg_data/` with:
  - Raw EEG data files (.fif, .set, .vhdr, etc.)
  - Channel information
  - Event markers
  - Task metadata
  - Dataset description

## Key Concepts

- **Orthographic Neighborhood Density**: The number of words that can be formed by changing one letter (Levenshtein distance = 1)
- **Levenshtein Distance**: A string metric for measuring the difference between two sequences
- **Median Split**: Words are categorized based on whether their density is above or below the median value
- **BIDS Format**: Brain Imaging Data Structure - a standard for organizing neuroimaging data
- **OpenNeuro**: A free and open platform for sharing neuroimaging data

## Example Output

```
HIGH DENSITY WORDS (neighbors >= median)
word: 15 neighbors
word: 14 neighbors
...

LOW DENSITY WORDS (neighbors < median)
word: 2 neighbors
word: 1 neighbors
...
```
