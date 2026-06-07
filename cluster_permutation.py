import logging
import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

RELATED_CODES = {'211', '212'}
UNRELATED_CODES = {'221', '222'}
TARGET_CODES = RELATED_CODES | UNRELATED_CODES

all_related_epochs = []
all_unrelated_epochs = []

for sub_dir in sorted(Path('data/derivatives/preprocessed_ica').glob('sub-*')):
    sub = sub_dir.name
    epo_file = sub_dir / f'{sub}_N400_ica-epo.fif'
    if not epo_file.exists():
        continue
    epochs = mne.read_epochs(str(epo_file), verbose=False)
    
    events, event_id = mne.events_from_annotations(epochs, verbose=False)
    target_id = {k: v for k, v in event_id.items() if str(k) in TARGET_CODES}
    if not target_id:
        continue
    
    related_keys = {k: v for k, v in target_id.items() if str(k) in RELATED_CODES}
    unrelated_keys = {k: v for k, v in target_id.items() if str(k) in UNRELATED_CODES}
    
    if related_keys:
        all_related_epochs.append(epochs[list(related_keys.keys())])
    if unrelated_keys:
        all_unrelated_epochs.append(epochs[list(unrelated_keys.keys())])
    
    logger.info(f'{sub}: {len(epochs)} epochs')

logger.info(f'Computing grand averages')
grand_related = mne.grand_average(all_related_epochs)
grand_unrelated = mne.grand_average(all_unrelated_epochs)

n400_chs = [c for c in grand_related.ch_names if c in ['Cz', 'CPz', 'Pz']]
if not n400_chs:
    n400_chs = grand_related.ch_names[:3]

logger.info(f'Using channels: {n400_chs}')

X = [grand_related.copy().pick(n400_chs).get_data(), grand_unrelated.copy().pick(n400_chs).get_data()]
X = [x.mean(axis=1) for x in X]  # Average across channels

logger.info(f'Running cluster permutation test (n=5000)')
from mne.stats import permutation_cluster_test

n_permutations = 5000
threshold = 2.0  # t-threshold for cluster formation

T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    X,
    n_permutations=n_permutations,
    threshold=threshold,
    tail=1,  # unrelated > related
    verbose=False
)

logger.info(f'Found {len(clusters)} significant clusters')
for i, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
    if p_val < 0.05:
        cluster_times = grand_related.times[cluster[0]]
        logger.info(f'Cluster {i+1}: {cluster_times[0]*1000:.0f}-{cluster_times[-1]*1000:.0f} ms, p={p_val:.4f}')

# Plot
times = grand_related.times * 1000
rel_data = grand_related.copy().pick(n400_chs).get_data().mean(axis=0) * 1e6
unrel_data = grand_unrelated.copy().pick(n400_chs).get_data().mean(axis=0) * 1e6

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times, rel_data, color='blue', linewidth=2, label='Related')
ax.plot(times, unrel_data, color='red', linewidth=2, label='Unrelated')
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')

# Shade significant clusters
for i, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
    if p_val < 0.05:
        cluster_times = grand_related.times[cluster[0]]
        ax.axvspan(cluster_times[0]*1000, cluster_times[-1]*1000, alpha=0.2, color='gray', label=f'Significant cluster (p={p_val:.4f})' if i == 0 else '')

ax.invert_yaxis()
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Amplitude (µV)')
ax.set_title(f'Grand Average ERP: Related vs Unrelated (Cz/CPz/Pz)\nCluster Permutation Test (n={n_permutations})')
ax.legend()
ax.set_xlim(-200, 800)
plt.tight_layout()
plt.savefig('figures/cluster_permutation_erp.png', dpi=300)
logger.info('Saved figures/cluster_permutation_erp.png')
