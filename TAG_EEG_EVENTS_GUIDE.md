# EEG Event Tagging Guide

## Overview

`tag_eeg_events.py` merges your orthographic neighborhood density baseline with experimental EEG timeline data to tag each word trial with its interference level (high or low).

## What It Does

1. **Loads density data**: Reads your pre-computed high/low density word lists
2. **Loads events**: Reads the BIDS `_events.tsv` file from your EEG dataset
3. **Maps words to trials**: Identifies which word was presented in each trial
4. **Calculates interference**: Tags each trial as 'high' or 'low' interference based on neighborhood density
5. **Saves results**: Outputs `tagged_events.csv` with the new tagging

## Current Status

The script was tested with the downloaded oddball dataset (ds003620), which contains:
- Trial types: "S  1" (standard stimuli) and "empty"
- **No actual words** - this is an auditory oddball task, not a word recognition task

Result: The script correctly identified that no words were present and created a template CSV file.

## Using with Word Recognition Datasets

To use this script with actual word recognition data:

### 1. Download a Word Recognition Dataset

Use `download_verbal_eeg.py` to download a dataset with word trials:
```bash
python download_verbal_eeg.py
```

Look for datasets like:
- ERP-CORE (ds000247) - N400 semantic processing task
- Lexical decision tasks
- Word recognition memory tasks

### 2. Run the Tagging Script

```bash
python tag_eeg_events.py
```

### 3. Expected Output

The script will:
- Load your density data (from `neighborhood_density_results.txt` or compute it)
- Find the events.tsv file in `data/`
- Extract words from the events (looks for columns like 'stimulus', 'word', 'text', etc.)
- Match each word to its density category
- Create `tagged_events.csv` with columns:
  - Original columns from events.tsv
  - `word`: The extracted word
  - `neighborhood_density`: Number of orthographic neighbors
  - `interference_level`: 'high' or 'low'

## Script Features

### Automatic Word Detection

The script looks for words in common column names:
- `stimulus`, `word`, `text`, `item`, `prime`, `target`, `condition`, `trial_type`

### Flexible Input

- If `neighborhood_density_results.txt` exists, it loads from there
- Otherwise, it computes density data from scratch using NLTK

### Error Handling

- Warns if no words are found in the events file
- Skips words not in the density list
- Handles various BIDS event file formats

## Example Output

When used with a proper word recognition dataset, the output would look like:

```
SUMMARY

Total trials: 500
Trials with words: 500
High interference trials: 250
Low interference trials: 250

Sample of tagged events:
 onset trial_type  word  neighborhood_density interference_level
  1000   word_cat   cat                    12               high
  2000   word_dog   dog                     3                low
  3000  word_bird  bird                     8               high
```

## Next Steps

1. **Download a word recognition dataset** from OpenNeuro
2. **Run the tagging script** to merge with your density data
3. **Use the tagged CSV** for EEG analysis (e.g., epoching by interference level)
4. **Compare brain responses** between high vs low interference conditions

## Troubleshooting

### "No words found in events file"
- Your dataset may not contain word trials (like the current oddball task)
- Check the events.tsv file to see what information is available
- Try a different dataset with explicit word stimuli

### "Word not in density list"
- The word may not be a 4-letter noun
- The word may not be in WordNet
- You can expand the density list to include more words

## Files Created

- `tagged_events.csv` - Main output with tagged events
- `neighborhood_density_results.txt` - Density data (if computed)

## Dependencies

- pandas
- nltk
- numpy

All listed in `requirements.txt`.
