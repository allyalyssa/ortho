import os
import pandas as pd
from pathlib import Path

def inspect_events():
    data_dir = Path("./data")
    
    # Recursively find the events.tsv file inside the data folder
    event_files = list(data_dir.rglob("*_events.tsv"))
    
    if not event_files:
        print("Could not find an events.tsv file. Double-check your data/ folder!")
        return
        
    tsv_path = event_files[0]
    print(f"Found timeline file at: {tsv_path}\n")
    
    # Load the TSV file using Pandas
    df = pd.read_csv(tsv_path, sep='\t')
    
    print("--- TIMELINE DATAFRAME COLUMNS ---")
    print(df.columns.tolist())
    
    print("\n--- FIRST 15 EXPERIMENTAL EVENTS ---")
    # Setting pandas display options to show all columns clearly
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head(15))
    
    # Let's count what kind of events are recorded
    if 'value' in df.columns:
        print("\n--- EVENT TRIGGER COUNTS ---")
        print(df['value'].value_counts())
    elif 'trial_type' in df.columns:
        print("\n--- EVENT TYPE COUNTS ---")
        print(df['trial_type'].value_counts())

if __name__ == "__main__":
    inspect_events()
