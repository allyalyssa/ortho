"""
Statistical Analysis Pipeline
Performs statistical analysis on ERP amplitudes in the N400 window (300-500ms).
"""

import logging
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)

def find_eeg_files() -> list[Path]:
    """Scan the data directory to find all EEG files."""
    data_dir = Path("./data/derivatives/preprocessed")
    eeg_files = list(data_dir.rglob("*_ica_eeg.fif"))
    
    if not eeg_files:
        logger.warning("No EEG files found in data directory")
        return []
    
    logger.info(f"Found {len(eeg_files)} EEG files")
    return eeg_files

def load_tagged_events() -> pd.DataFrame | None:
    """Load the tagged events CSV if it exists."""
    tagged_file = Path("tagged_events.csv")
    if tagged_file.exists():
        df = pd.read_csv(tagged_file)
        logger.info(f"Loaded tagged events: {len(df)} trials")
        return df
    return None

def process_single_subject(eeg_file: Path, tagged_events: pd.DataFrame | None = None, single_trial_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Process a single subject's EEG data and extract single-trial N400 amplitudes."""
    
    if single_trial_df is None:
        single_trial_df = pd.DataFrame(columns=['Subject', 'Trial_Type', 'N400_Amplitude'])
    
    try:
        raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
        logger.info(f"Loaded {len(raw.ch_names)} channels, {raw.n_times} timepoints")
    except (ValueError, RuntimeError, OSError) as e:
        logger.warning(f"Skipping corrupted file {eeg_file.name}: {e}")
        return single_trial_df
    
    raw.filter(l_freq=0.1, h_freq=20, verbose=False)
    raw_filtered = raw
    
    if tagged_events is not None:
        logger.info(f"Using tagged events from CSV for {eeg_file.name}")
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
        logger.warning("No tagged events found")
        return single_trial_df
    
    logger.info(f"Creating epochs with baseline correction and artifact rejection for {eeg_file.name}")
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
    
    epochs.apply_baseline((-0.2, 0))
    logger.info(f"Created {len(epochs)} trials (after artifact rejection)")
    
    if 'Pz' not in epochs.ch_names:
        logger.warning("Pz channel not found, using first available channel")
        target_channel = epochs.ch_names[0]
    else:
        target_channel = 'Pz'
    
    logger.info("Extracting single-trial N400 amplitudes...")
    ch_idx = epochs.ch_names.index(target_channel)
    time_mask = (epochs.times >= 0.3) & (epochs.times <= 0.5)
    
    # Collect trial data in a list
    trial_data = []
    
    # Loop through each condition and each trial
    for condition in ['high_interference', 'low_interference']:
        if condition in epochs.event_id:
            condition_epochs = epochs[condition]
            logger.info(f"Processing {condition}: {len(condition_epochs)} trials")
            
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
    
    logger.info(f"Total trials added: {len(single_trial_df)}")
    return single_trial_df

def report_significance(p_value: float) -> None:
    """Report statistical significance based on p-value."""
    if p_value < 0.05:
        logger.info(f"Result: SIGNIFICANT effect (p < 0.05)")
    elif p_value < 0.10:
        logger.info(f"Result: Trend toward significance (p < 0.10)")
    else:
        logger.info(f"Result: No significant effect (p >= 0.05)")

def run_statistical_analysis(single_trial_df: pd.DataFrame) -> pd.DataFrame:
    single_trial_df = single_trial_df.copy()
    
    # Ensure N400_Amplitude is float first
    single_trial_df['N400_Amplitude'] = pd.to_numeric(single_trial_df['N400_Amplitude'], errors='coerce')
    
    # Remove NaN values
    single_trial_df = single_trial_df.dropna(subset=['N400_Amplitude'])
    
    # Remove infinite values
    single_trial_df = single_trial_df[np.isfinite(single_trial_df['N400_Amplitude'])]
    
    logger.info(f"Total trials after cleaning: {len(single_trial_df)}")
    
    # Separate by condition for summary statistics
    high_trials = single_trial_df[single_trial_df['Trial_Type'] == 'high_interference']['N400_Amplitude'].values
    low_trials = single_trial_df[single_trial_df['Trial_Type'] == 'low_interference']['N400_Amplitude'].values
    
    logger.info(f"High interference trials: {len(high_trials)}")
    logger.info(f"Low interference trials: {len(low_trials)}")
    logger.info(f"High interference mean: {high_trials.mean():.2f} uV (SD: {high_trials.std():.2f})")
    logger.info(f"Low interference mean: {low_trials.mean():.2f} uV (SD: {low_trials.std():.2f})")
    
    logger.info("Fitting Linear Mixed-Effects Model")
    logger.info("Formula: N400_Amplitude ~ Trial_Type")
    logger.info("Random effect: Subject (intercept)")
    
    try:
        model = smf.mixedlm("N400_Amplitude ~ Trial_Type", single_trial_df, groups=single_trial_df["Subject"])
        result = model.fit()
        
        logger.info("\n" + "=" * 70)
        logger.info("MODEL SUMMARY")
        logger.info("=" * 70)
        logger.info(str(result.summary()))
        
        # Extract fixed effect p-value for Trial_Type
        if 'Trial_Type' in result.pvalues:
            trial_type_p = result.pvalues['Trial_Type[T.low_interference]']
            logger.info(f"FIXED EFFECT: Trial_Type p-value: {trial_type_p:.4f}")
            report_significance(trial_type_p)
        
    except Exception as e:
        logger.error(f"Error fitting LME model: {e}")
        logger.info("Falling back to simple t-test...")
        
        t_stat, t_p = stats.ttest_ind(high_trials, low_trials)
        logger.info(f"Independent t-test: t = {t_stat:.4f}, p = {t_p:.4f}")
        report_significance(t_p)
    
    logger.info(f"Total N (trials): {len(single_trial_df)}")
    logger.info(f"High interference: {len(high_trials)} trials, mean = {high_trials.mean():.2f} uV")
    logger.info(f"Low interference: {len(low_trials)} trials, mean = {low_trials.mean():.2f} uV")
    logger.info(f"Mean difference: {high_trials.mean() - low_trials.mean():.2f} uV")
    
    return single_trial_df

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    
    eeg_files = find_eeg_files()
    
    if not eeg_files:
        logger.warning("No EEG files found. Exiting.")
        return
    
    tagged_events = load_tagged_events()
    single_trial_df = pd.DataFrame(columns=['Subject', 'Trial_Type', 'N400_Amplitude'])
    skipped_files = 0
    
    for eeg_file in eeg_files:
        single_trial_df = process_single_subject(eeg_file, tagged_events, single_trial_df)
        if len(single_trial_df) == 0:
            skipped_files += 1
    
    logger.info(f"Files successfully processed: {len(eeg_files) - skipped_files}")
    logger.info(f"Files skipped (corrupted): {skipped_files}")
    
    if len(single_trial_df) >= 2:
        run_statistical_analysis(single_trial_df)
    else:
        logger.warning(f"Not enough trials for statistical analysis (need at least 2, got {len(single_trial_df)})")

if __name__ == "__main__":
    main()
