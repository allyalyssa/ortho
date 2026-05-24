"""
Statistical Analysis Pipeline
Performs statistical analysis on ERP amplitudes in the N400 window (300-500ms).
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf

def find_eeg_files():
    """
    Scan the data directory to find all EEG files.
    """
    print("Scanning data directory for EEG files...")
    data_dir = Path("./data/derivatives/preprocessed")
    
    # Find all .fif files in the preprocessed directory
    eeg_files = list(data_dir.rglob("*_ica_eeg.fif"))
    
    if not eeg_files:
        print("No EEG files found in data directory")
        return []
    
    print(f"Found {len(eeg_files)} EEG files:")
    for f in eeg_files:
        print(f"  - {f.name}")
    
    return eeg_files

def load_tagged_events():
    """
    Load the tagged events CSV if it exists.
    """
    tagged_file = Path("tagged_events.csv")
    if tagged_file.exists():
        df = pd.read_csv(tagged_file)
        print(f"Loaded tagged events: {len(df)} trials")
        return df
    return None

def process_single_subject(eeg_file, tagged_events=None, single_trial_df=None):
    """
    Process a single subject's EEG data and extract single-trial N400 amplitudes.
    """
    print(f"\nProcessing: {eeg_file.name}")
    
    try:
        # Load raw data
        print("  Loading raw data...")
        raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
        print(f"  Loaded: {len(raw.ch_names)} channels, {raw.n_times} timepoints")
    except (ValueError, RuntimeError, Exception) as e:
        print(f"  Skipping corrupted file: {e}")
        return single_trial_df
    
    # Apply bandpass filter (in-place to avoid memory issues)
    print("  Applying bandpass filter (0.1 - 20 Hz)...")
    raw.filter(l_freq=0.1, h_freq=20, verbose=False)
    raw_filtered = raw
    
    # Load events
    if tagged_events is not None:
        print("  Using tagged events from CSV...")
        events_df = tagged_events.copy()
        events_df = events_df[events_df['interference_level'].notna()]
        
        # Convert onset to sample indices (onset is in milliseconds, convert to seconds)
        sfreq = raw_filtered.info['sfreq']
        events_df['sample'] = (events_df['onset'] / 1000 * sfreq).astype(int)
        
        # Create event array - map 'high'/'low' to event IDs
        event_id_map = {'high': 1, 'low': 2}
        events_df['event_id'] = events_df['interference_level'].map(event_id_map)
        
        events_array = []
        for idx, row in events_df.iterrows():
            events_array.append([row['sample'], 0, row['event_id']])
        events_array = np.array(events_array, dtype=int)
        
        event_id_dict = {'high_interference': 1, 'low_interference': 2}
    else:
        print("  No tagged events found")
        return single_trial_df
    
    # Create epochs with baseline correction and artifact rejection
    print("  Creating epochs with baseline correction and artifact rejection...")
    reject = dict(eeg=100e-6)  # Reject epochs with amplitude > 100 microvolts
    
    epochs = mne.Epochs(
        raw_filtered,
        events_array,
        event_id=event_id_dict,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0),  # Explicit baseline correction
        reject=reject,
        preload=True,
        verbose=False
    )
    
    # Explicitly apply baseline correction
    epochs.apply_baseline((-0.2, 0))
    
    print(f"  Created epochs: {len(epochs)} trials (after artifact rejection)")
    
    # Check if Pz channel exists
    if 'Pz' not in epochs.ch_names:
        print("  Pz channel not found, using first available channel")
        target_channel = epochs.ch_names[0]
    else:
        target_channel = 'Pz'
    
    # Extract single-trial amplitudes
    print("  Extracting single-trial N400 amplitudes...")
    ch_idx = epochs.ch_names.index(target_channel)
    time_mask = (epochs.times >= 0.3) & (epochs.times <= 0.5)
    
    # Collect trial data in a list
    trial_data = []
    
    # Loop through each condition and each trial
    for condition in ['high_interference', 'low_interference']:
        if condition in epochs.event_id:
            condition_epochs = epochs[condition]
            print(f"  Processing {condition}: {len(condition_epochs)} trials")
            
            for trial_idx in range(len(condition_epochs)):
                # Get single trial data
                trial_data_single = condition_epochs.get_data()[trial_idx, ch_idx, :]
                
                # Calculate mean amplitude in N400 window
                n400_amp_uv = np.mean(trial_data_single[time_mask]) * 1e6
                
                # Append to trial data list
                trial_data.append({
                    'Subject': eeg_file.name,
                    'Trial_Type': condition,
                    'N400_Amplitude': n400_amp_uv
                })
    
    # Concatenate all trial data at once
    if trial_data:
        new_trials_df = pd.DataFrame(trial_data)
        single_trial_df = pd.concat([single_trial_df, new_trials_df], ignore_index=True)
    
    print(f"  Total trials added: {len(single_trial_df)}")
    return single_trial_df

def run_statistical_analysis(single_trial_df):
    """
    Run statistical tests on the single-trial data using Linear Mixed-Effects Model.
    """
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS (Linear Mixed-Effects Model)")
    print("=" * 70)
    
    # Clean DataFrame - remove NaN or infinite values
    print("\nCleaning data...")
    single_trial_df = single_trial_df.copy()
    
    # Ensure N400_Amplitude is float first
    single_trial_df['N400_Amplitude'] = pd.to_numeric(single_trial_df['N400_Amplitude'], errors='coerce')
    
    # Remove NaN values
    single_trial_df = single_trial_df.dropna(subset=['N400_Amplitude'])
    
    # Remove infinite values
    single_trial_df = single_trial_df[np.isfinite(single_trial_df['N400_Amplitude'])]
    
    print(f"Total trials after cleaning: {len(single_trial_df)}")
    
    # Separate by condition for summary statistics
    high_trials = single_trial_df[single_trial_df['Trial_Type'] == 'high_interference']['N400_Amplitude'].values
    low_trials = single_trial_df[single_trial_df['Trial_Type'] == 'low_interference']['N400_Amplitude'].values
    
    print(f"High interference trials: {len(high_trials)}")
    print(f"Low interference trials: {len(low_trials)}")
    
    print(f"\nHigh interference mean: {high_trials.mean():.2f} uV (SD: {high_trials.std():.2f})")
    print(f"Low interference mean: {low_trials.mean():.2f} uV (SD: {low_trials.std():.2f})")
    
    # Fit Linear Mixed-Effects Model
    print("\n" + "-" * 70)
    print("Linear Mixed-Effects Model")
    print("-" * 70)
    print("Formula: N400_Amplitude ~ Trial_Type")
    print("Random effect: Subject (intercept)")
    
    try:
        model = smf.mixedlm("N400_Amplitude ~ Trial_Type", single_trial_df, groups=single_trial_df["Subject"])
        result = model.fit()
        
        print("\n" + "=" * 70)
        print("MODEL SUMMARY")
        print("=" * 70)
        print(result.summary())
        
        # Extract fixed effect p-value for Trial_Type
        if 'Trial_Type' in result.pvalues:
            trial_type_p = result.pvalues['Trial_Type[T.low_interference]']
            print("\n" + "-" * 70)
            print("FIXED EFFECT: Trial_Type")
            print("-" * 70)
            print(f"p-value: {trial_type_p:.4f}")
            
            if trial_type_p < 0.05:
                print(f"\nResult: SIGNIFICANT effect of Trial_Type (p < 0.05)")
            elif trial_type_p < 0.10:
                print(f"\nResult: Trend toward significance (p < 0.10)")
            else:
                print(f"\nResult: No significant effect (p >= 0.05)")
        
    except Exception as e:
        print(f"\nError fitting LME model: {e}")
        print("Falling back to simple t-test...")
        
        # Fallback to t-test if LME fails
        t_stat, t_p = stats.ttest_ind(high_trials, low_trials)
        print(f"\nIndependent t-test: t = {t_stat:.4f}, p = {t_p:.4f}")
        
        if t_p < 0.05:
            print(f"\nResult: SIGNIFICANT difference (p < 0.05)")
        elif t_p < 0.10:
            print(f"\nResult: Trend toward significance (p < 0.10)")
        else:
            print(f"\nResult: No significant difference (p >= 0.05)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total N (trials): {len(single_trial_df)}")
    print(f"High interference: {len(high_trials)} trials, mean = {high_trials.mean():.2f} uV")
    print(f"Low interference: {len(low_trials)} trials, mean = {low_trials.mean():.2f} uV")
    print(f"Mean difference: {high_trials.mean() - low_trials.mean():.2f} uV")
    
    return single_trial_df

def main():
    """
    Main statistical analysis pipeline.
    """
    print("=" * 70)
    print("STATISTICAL ANALYSIS PIPELINE (Single-Trial Level)")
    print("=" * 70)
    
    # Step 1: Find EEG files
    eeg_files = find_eeg_files()
    
    if not eeg_files:
        print("No EEG files found. Exiting.")
        return
    
    # Step 2: Load tagged events
    tagged_events = load_tagged_events()
    
    # Step 3: Initialize master DataFrame for single-trial data
    single_trial_df = pd.DataFrame(columns=['Subject', 'Trial_Type', 'N400_Amplitude'])
    
    # Step 4: Process each file
    skipped_files = 0
    
    for eeg_file in eeg_files:
        single_trial_df = process_single_subject(eeg_file, tagged_events, single_trial_df)
        
        if len(single_trial_df) == 0:
            skipped_files += 1
    
    print(f"\nFiles successfully processed: {len(eeg_files) - skipped_files}")
    print(f"Files skipped (corrupted): {skipped_files}")
    
    # Step 5: Run statistical analysis
    if len(single_trial_df) >= 2:
        run_statistical_analysis(single_trial_df)
    else:
        print(f"\nNot enough trials for statistical analysis (need at least 2, got {len(single_trial_df)})")

if __name__ == "__main__":
    main()
