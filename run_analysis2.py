import mne
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf

preprocessed_dir = Path('data/derivatives/preprocessed')
epoch_files = sorted(preprocessed_dir.glob('sub-*_N400_preprocessed-epo.fif'))

RELATED_VALUES = {112, 122}
UNRELATED_VALUES = {212, 222}

rows = []
for f in epoch_files:
    sub = f.name.split('_')[0]
    epochs = mne.read_epochs(f, preload=True, verbose=False)
    if len(epochs) < 30:
        continue
    n400_chs = [c for c in epochs.ch_names if c in ['Cz', 'CPz', 'Pz']]
    if not n400_chs:
        n400_chs = epochs.ch_names[:3]
    data = epochs.copy().crop(tmin=0.300, tmax=0.500).get_data(picks=n400_chs)
    amp = data.mean(axis=(1, 2)) * 1e6
    events_tsv = Path(f'data/erpcore/N400/{sub}/eeg/{sub}_task-N400_events.tsv')
    ev = pd.read_csv(events_tsv, sep='\t')
    targets = ev[ev['value'].isin(RELATED_VALUES | UNRELATED_VALUES)].copy()
    targets = targets.reset_index(drop=True)
    targets = targets.iloc[:len(amp)]
    for i, row in targets.iterrows():
        if i >= len(amp):
            break
        val = row['value']
        if val in RELATED_VALUES:
            condition = 'related'
        else:
            condition = 'unrelated'
        rows.append({'Subject': sub, 'N400_Amplitude': amp[i], 'Condition': condition})

df = pd.DataFrame(rows)
print(df['Condition'].value_counts())
print(f'Total trials: {len(df)}, Subjects: {df["Subject"].nunique()}')

related = df[df['Condition'] == 'related']['N400_Amplitude']
unrelated = df[df['Condition'] == 'unrelated']['N400_Amplitude']
print(f'Related mean: {related.mean():.3f} uV')
print(f'Unrelated mean: {unrelated.mean():.3f} uV')

model = smf.mixedlm('N400_Amplitude ~ Condition', df, groups=df['Subject'])
result = model.fit()
print(result.summary())
