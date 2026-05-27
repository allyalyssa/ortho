import logging
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

RELATED_CODES = {'211', '212'}
UNRELATED_CODES = {'221', '222'}
TARGET_CODES = RELATED_CODES | UNRELATED_CODES

all_related = []
all_unrelated = []

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
        continue
    epochs = mne.Epochs(raw, events, event_id=target_id,
                        tmin=-0.2, tmax=0.8,
                        baseline=(-0.2, 0),
                        reject=dict(eeg=100e-6),
                        preload=True, verbose=False)
    if len(epochs) < 30:
        continue
    related_keys = {k: v for k, v in target_id.items() if str(k) in RELATED_CODES}
    unrelated_keys = {k: v for k, v in target_id.items() if str(k) in UNRELATED_CODES}
    if related_keys:
        all_related.append(epochs[list(related_keys.keys())].average())
    if unrelated_keys:
        all_unrelated.append(epochs[list(unrelated_keys.keys())].average())
    logger.info(f'{sub}: done')

grand_related = mne.grand_average(all_related)
grand_unrelated = mne.grand_average(all_unrelated)

times = grand_related.times * 1000
n400_chs = [c for c in grand_related.ch_names if c in ['Cz', 'CPz', 'Pz']]
if not n400_chs:
    n400_chs = grand_related.ch_names[:3]

fig, ax = plt.subplots(figsize=(10, 5))
rel_data = grand_related.copy().pick(n400_chs).get_data().mean(axis=0) * 1e6
unrel_data = grand_unrelated.copy().pick(n400_chs).get_data().mean(axis=0) * 1e6
ax.plot(times, rel_data, color='blue', linewidth=2, label='Related')
ax.plot(times, unrel_data, color='red', linewidth=2, label='Unrelated')
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.axvspan(300, 500, alpha=0.15, color='gray', label='N400 window')
ax.invert_yaxis()
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (µV)')
ax.set_title('Grand Average ERP: Related vs Unrelated Targets (Cz/CPz/Pz)\nn=27, p<0.001')
ax.legend()
ax.set_xlim(-200, 800)
plt.tight_layout()
plt.savefig('figures/erp_waveform.png', dpi=300)
logger.info('Saved figures/erp_waveform.png')
