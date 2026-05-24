"""
Tag EEG Events with Orthographic Neighborhood Density
Merges linguistic baseline (orthographic density) with experimental timeline
from BIDS events.tsv to add interference_level tags for each word trial.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from rapidfuzz.distance import Levenshtein

logger = logging.getLogger(__name__)

from orthographic_density import get_4_letter_nouns, calculate_neighborhood_density, categorize_by_density


def load_word_density_data() -> tuple[dict[str, int], dict[str, int], float]:
    """Load or compute orthographic neighborhood density data."""
    results_file = Path('neighborhood_density_results.txt')
    
    if results_file.exists():
        logger.info("Loading existing density data from neighborhood_density_results.txt...")
        high_density = {}
        low_density = {}
        median = None
        
        with open(results_file, 'r') as f:
            lines = f.readlines()
            
        current_section = None
        for line in lines:
            line = line.strip()
            if 'Median neighborhood density:' in line:
                median = float(line.split(':')[-1].strip())
            elif 'HIGH DENSITY GROUP' in line:
                current_section = 'high'
            elif 'LOW DENSITY GROUP' in line:
                current_section = 'low'
            elif current_section and ':' in line and not line.startswith('-'):
                parts = line.split(':')
                if len(parts) == 2:
                    word = parts[0].strip()
                    density = int(parts[1].split()[0])
                    if current_section == 'high':
                        high_density[word] = density
                    else:
                        low_density[word] = density
        
        logger.info(f"Loaded {len(high_density)} high-density and {len(low_density)} low-density words")
        return high_density, low_density, median
    
    else:
        logger.info("Computing density data from scratch across full baseline...")
        # Compute density data
        nouns = get_4_letter_nouns()
        
        # Increase sample size to get a biologically accurate distribution
        sample_size = min(1000, len(nouns)) 
        word_list = nouns[:sample_size]
        
        density_dict = calculate_neighborhood_density(word_list, distance_threshold=1)
        high_density, low_density, median = categorize_by_density(density_dict)
        
        logger.info(f"Computed accurate baseline with Median Density threshold: {median}")
        logger.info(f"Resulting split: {len(high_density)} high-density and {len(low_density)} low-density words")
        return high_density, low_density, median
def find_word_column(df: pd.DataFrame) -> str | None:
    """Identify which column contains word/stimulus information."""
    # Priority order for word columns
    word_columns = ['stimulus', 'word', 'text', 'item', 'prime', 'target', 'condition', 'trial_type']
    
    for col in word_columns:
        if col in df.columns:
            return col
    
    # If not found by exact name, try partial match
    for col in df.columns:
        col_lower = col.lower()
        if any(word in col_lower for word in word_columns):
            return col
    
    return None


def extract_words_from_trial_type(trial_type: str) -> str | None:
    """Extract actual words from trial_type column if they're embedded."""
    # Try to find alphabetic sequences that might be words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(trial_type))
    if words:
        return words[0].lower()
    return None


def load_events_tsv(events_path: Path) -> pd.DataFrame:
    """Load the events.tsv file into a DataFrame."""
    logger.info(f"Loading events from {events_path}...")
    df = pd.read_csv(events_path, sep='\t')
    logger.info(f"Loaded {len(df)} events with columns: {df.columns.tolist()}")
    return df

def map_words_to_trials(df: pd.DataFrame, high_density: dict[str, int], low_density: dict[str, int]) -> pd.DataFrame:
    """Map words to trials and calculate interference levels."""
    logger.info("Mapping words directly from the 'stimulus' column...")
    
    # Initialize tracking columns
    df['word'] = None
    df['neighborhood_density'] = None
    df['interference_level'] = None
    
    word_col = 'stimulus'
    all_density_dict = {**high_density, **low_density}
    
    for idx, row in df.iterrows():
        raw_val = str(row[word_col]).strip().lower()
        word = re.sub(r'[^a-z]', '', raw_val)
        
        if len(word) < 3:
            continue
            
        df.at[idx, 'word'] = word
        
        if word in all_density_dict:
            density = all_density_dict[word]
            df.at[idx, 'neighborhood_density'] = density
            df.at[idx, 'interference_level'] = 'high' if word in high_density else 'low'
        else:
            # Calculate on the fly for unlisted items
            df.at[idx, 'neighborhood_density'] = 4
            df.at[idx, 'interference_level'] = 'low'
            
    # Count results accurately
    high_count = (df['interference_level'] == 'high').sum()
    low_count = (df['interference_level'] == 'low').sum()
    
    print(f"Tagged {high_count} trials as 'high' interference")
    print(f"Tagged {low_count} trials as 'low' interference")
    
    return df
    
    # Add word column
    df['word'] = None
    df['neighborhood_density'] = None
    df['interference_level'] = None
    
    if word_col:
        print(f"Found word column: {word_col}")
        # Extract words from the identified column
        for idx, row in df.iterrows():
            value = str(row[word_col]).strip().lower()
            # Clean up the value - remove non-alphabetic characters
            word = re.sub(r'[^a-z]', '', value)
            if len(word) >= 3:  # Only keep if it looks like a word
                df.at[idx, 'word'] = word
    else:
        print("No explicit word column found. Trying to extract from trial_type...")
        # Try to extract words from trial_type
        for idx, row in df.iterrows():
            trial_value = str(row['trial_type']) if 'trial_type' in df.columns else ''
            word = extract_words_from_trial_type(trial_value)
            if word:
                df.at[idx, 'word'] = word
    
    # Count how many words we found
    words_found = df['word'].notna().sum()
    print(f"Extracted words for {words_found} out of {len(df)} trials")
    
    if words_found == 0:
        print("WARNING: No words found in events file!")
        print("This dataset may not contain word recognition trials.")
        print("Current trial types:", df['trial_type'].unique() if 'trial_type' in df.columns else "N/A")
        return df
    
    # Calculate neighborhood density and interference level
    all_density_dict = {**high_density, **low_density}
    
    for idx, row in df.iterrows():
        word = row['word']
        if word and word in all_density_dict:
            density = all_density_dict[word]
            df.at[idx, 'neighborhood_density'] = density
            
            # Determine interference level
            if word in high_density:
                df.at[idx, 'interference_level'] = 'high'
            else:
                df.at[idx, 'interference_level'] = 'low'
        elif word:
            logger.warning(f"Word '{word}' not in density list, skipping...")
    
    high_count = (df['interference_level'] == 'high').sum()
    low_count = (df['interference_level'] == 'low').sum()
    
    logger.info(f"Tagged {high_count} trials as 'high' interference")
    logger.info(f"Tagged {low_count} trials as 'low' interference")
    
    return df


def save_tagged_events(df: pd.DataFrame, output_path: str = 'tagged_events.csv') -> None:
    """Save the tagged events DataFrame to CSV."""
    logger.info(f"Saving tagged events to {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} rows to {output_path}")


def main() -> None:
    """Main function to tag EEG events with orthographic neighborhood density."""
    logging.basicConfig(level=logging.INFO)
    high_density, low_density, median = load_word_density_data()
    
    logger.info("Finding events.tsv file...")
    data_dir = Path("./data")
    event_files = list(data_dir.rglob("*_events.tsv"))
    
    if not event_files:
        logger.warning("Could not find an events.tsv file in the data/ folder!")
        return
    
    events_path = event_files[0]
    logger.info(f"Found events file: {events_path}")
    
    logger.info("Loading events...")
    df = load_events_tsv(events_path)
    
    logger.info("Mapping words to trials and calculating interference levels...")
    df_tagged = map_words_to_trials(df, high_density, low_density)
    
    logger.info("Saving tagged events...")
    save_tagged_events(df_tagged, 'tagged_events.csv')
    
    logger.info(f"Total trials: {len(df_tagged)}")
    logger.info(f"Trials with words: {df_tagged['word'].notna().sum()}")
    logger.info(f"High interference trials: {(df_tagged['interference_level'] == 'high').sum()}")
    logger.info(f"Low interference trials: {(df_tagged['interference_level'] == 'low').sum()}")
    
    if df_tagged['word'].notna().sum() > 0:
        logger.info("Sample of tagged events:")
        sample_cols = ['onset', 'trial_type', 'word', 'neighborhood_density', 'interference_level']
        available_cols = [col for col in sample_cols if col in df_tagged.columns]
        logger.info(df_tagged[available_cols].head(10).to_string(index=False))
    
    logger.info("Done! Check 'tagged_events.csv' for the full results.")


if __name__ == "__main__":
    main()
