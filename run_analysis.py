import mne
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf

preprocessed_dir = Path('data/derivatives/preprocessed')
epoch_files = sorted(preprocessed_dir.glob('sub-*_N400_preprocessed-epo.fif'))
print(f'Found {len(epoch_files)} epoch files')

rows = []
for f in epoch_files:
    sub = f.name.split('_')[0]
    epochs = mne.read_epochs(f, preload=True, verbose=False)
    if len(epochs) < 30:
        print(f'Excluding {sub}: only {len(epochs)} epochs')
        continue
    n400_chs = [c for c in epochs.ch_names if c in ['Cz', 'CPz', 'Pz']]
    if not n400_chs:
        n400_chs = epochs.ch_names[:3]
    data = epochs.copy().crop(tmin=0.300, tmax=0.500).get_data(picks=n400_chs)
    amp = data.mean(axis=(1, 2)) * 1e6
    events_tsv = Path(f'data/erpcore/N400/{sub}/eeg/{sub}_task-N400_events.tsv')
    ev = pd.read_csv(events_tsv, sep='\t')
    trial_types = ev[ev['trial_type'].notna()]['trial_type'].values
    trial_types = trial_types[:len(amp)]
    for a, t in zip(amp, trial_types):
        rows.append({'Subject': sub, 'N400_Amplitude': a, 'Trial_Type': t})

df = pd.DataFrame(rows)
print(df['Trial_Type'].value_counts())
print(f'Total trials: {len(df)}, Subjects: {df["Subject"].nunique()}')

model = smf.mixedlm('N400_Amplitude ~ Trial_Type', df, groups=df['Subject'])
result = model.fit()
print(result.summary())
