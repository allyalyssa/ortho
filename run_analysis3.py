import mne
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf

RELATED_CODES = {'111', '112', '121', '122'}
UNRELATED_CODES = {'211', '212', '221', '222'}
TARGET_CODES = RELATED_CODES | UNRELATED_CODES

rows = []
for sub_dir in sorted(Path('data/erpcore/N400').glob('sub-*')):
    sub = sub_dir.name
    set_file = sub_dir / 'eeg' / f'{sub}_task-N400_eeg.set'
    if not set_file.exists():
        continue
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    raw.filter(0.1, 20.0, fir_design='firwin', verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_id = {k: v for k, v in event_id.items() if str(k) in TARGET_CODES}
    if not target_id:
        print(f'{sub}: no target events found, skipping')
        continue
    reverse_id = {v: str(k) for k, v in target_id.items()}
    epochs = mne.Epochs(raw, events, event_id=target_id,
                        tmin=-0.2, tmax=0.8,
                        baseline=(-0.2, 0),
                        reject=dict(eeg=100e-6),
                        preload=True, verbose=False)
    if len(epochs) < 30:
        print(f'Excluding {sub}: only {len(epochs)} epochs')
        continue
    n400_chs = [c for c in epochs.ch_names if c in ['Cz', 'CPz', 'Pz']]
    if not n400_chs:
        n400_chs = epochs.ch_names[:3]
    data = epochs.copy().crop(tmin=0.3, tmax=0.5).get_data(picks=n400_chs)
    amp = data.mean(axis=(1, 2)) * 1e6
    for i, event in enumerate(epochs.events):
        code_str = reverse_id[event[2]]
        condition = 'related' if code_str in RELATED_CODES else 'unrelated'
        rows.append({'Subject': sub, 'N400_Amplitude': amp[i], 'Condition': condition})
    print(f'{sub}: {len(epochs)} epochs')

df = pd.DataFrame(rows)
print(df['Condition'].value_counts())
print(f'Total trials: {len(df)}, Subjects: {df["Subject"].nunique()}')
print(f'Related mean: {df[df.Condition=="related"]["N400_Amplitude"].mean():.3f} uV')
print(f'Unrelated mean: {df[df.Condition=="unrelated"]["N400_Amplitude"].mean():.3f} uV')

model = smf.mixedlm('N400_Amplitude ~ Condition', df, groups=df['Subject'])
result = model.fit()
print(result.summary())
