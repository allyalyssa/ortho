import logging
import mne
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from tqdm import tqdm

logger = logging.getLogger(__name__)

RELATED_CODES = {'211', '212'}
UNRELATED_CODES = {'221', '222'}
TARGET_CODES = RELATED_CODES | UNRELATED_CODES


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    density_df = pd.read_csv('data/stimuli_density.csv')
    event_code_to_density = dict(zip(density_df['event_code'], density_df['density']))
    
    rows = []
    for sub_dir in sorted(Path('data/derivatives/preprocessed_ica').glob('sub-*')):
        sub = sub_dir.name
        epo_file = sub_dir / f'{sub}_N400_ica-epo.fif'
        if not epo_file.exists():
            continue
        epochs = mne.read_epochs(str(epo_file), verbose=False)
        
        events, event_id = mne.events_from_annotations(epochs, verbose=False)
        target_id = {k: v for k, v in event_id.items() if str(k) in TARGET_CODES}
        if not target_id:
            logger.warning(f'{sub}: no target events found, skipping')
            continue
        reverse_id = {v: str(k) for k, v in target_id.items()}
        
        if len(epochs) < 30:
            logger.warning(f'Excluding {sub}: only {len(epochs)} epochs')
            continue
        n400_chs = [c for c in epochs.ch_names if c in ['Cz', 'CPz', 'Pz']]
        if not n400_chs:
            n400_chs = epochs.ch_names[:3]
        data = epochs.copy().crop(tmin=0.3, tmax=0.5).get_data(picks=n400_chs)
        amp = data.mean(axis=(1, 2)) * 1e6
        for i, event in enumerate(epochs.events):
            code_str = reverse_id[event[2]]
            condition = 'related' if code_str in RELATED_CODES else 'unrelated'
            density = event_code_to_density.get(code_str, 0)
            rows.append({'Subject': sub, 'N400_Amplitude': amp[i], 'Condition': condition, 'density': density, 'event_code': code_str})
        logger.info(f'{sub}: {len(epochs)} epochs')
    
    df = pd.DataFrame(rows)
    logger.info(f'Total trials: {len(df)}, Subjects: {df["Subject"].nunique()}')
    
    # Model 1: main effects
    logger.info('--- Model 1: Condition + density ---')
    m1 = smf.mixedlm('N400_Amplitude ~ Condition + density', df, groups=df['Subject']).fit()
    for line in str(m1.summary()).split('\n'):
        logger.info(line)
    
    # Model 2: interaction
    logger.info('--- Model 2: Condition * density ---')
    m2 = smf.mixedlm('N400_Amplitude ~ Condition * density', df, groups=df['Subject']).fit()
    for line in str(m2.summary()).split('\n'):
        logger.info(line)
    
    # Model 3: permutation test on density coefficient
    logger.info('--- Permutation test on density (n=2000) ---')
    observed_beta = m1.fe_params['density']
    null_betas = []
    rng = np.random.default_rng(42)
    for _ in tqdm(range(2000), desc='Permutation test'):
        df['density_perm'] = rng.permutation(df['density'].values)
        m_perm = smf.mixedlm('N400_Amplitude ~ Condition + density_perm', df, groups=df['Subject']).fit(disp=False)
        null_betas.append(m_perm.fe_params['density_perm'])
    null_betas = np.array(null_betas)
    perm_p = np.mean(np.abs(null_betas) >= np.abs(observed_beta))
    logger.info(f'Observed density beta: {observed_beta:.3f}')
    logger.info(f'Permutation p-value: {perm_p:.4f}')
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_betas, bins=50, color='gray', alpha=0.7, label='Null distribution')
    ax.axvline(observed_beta, color='red', linewidth=2, label=f'Observed beta={observed_beta:.3f}')
    ax.set_xlabel('Density coefficient')
    ax.set_ylabel('Count')
    ax.set_title(f'Permutation test: density effect (permutation p={perm_p:.4f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/permutation_density.png', dpi=300)
    logger.info('Saved figures/permutation_density.png')
    
    # Cluster permutation test on time axis
    logger.info('--- Cluster permutation test on time axis (n=5000) ---')
    
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
    
    if all_related_epochs and all_unrelated_epochs:
        grand_related = mne.grand_average(all_related_epochs)
        grand_unrelated = mne.grand_average(all_unrelated_epochs)
        
        n400_chs = [c for c in grand_related.ch_names if c in ['Cz', 'CPz', 'Pz']]
        if not n400_chs:
            n400_chs = grand_related.ch_names[:3]
        
        logger.info(f'Using channels: {n400_chs}')
        
        X = [grand_related.copy().pick(n400_chs).get_data(), grand_unrelated.copy().pick(n400_chs).get_data()]
        X = [x.mean(axis=1) for x in X]
        
        from mne.stats import permutation_cluster_test
        
        n_permutations = 5000
        threshold = 2.0
        
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            X,
            n_permutations=n_permutations,
            threshold=threshold,
            tail=1,
            verbose=False
        )
        
        logger.info(f'Found {len(clusters)} clusters')
        for i, (cluster, p_val) in enumerate(zip(clusters, cluster_p_values)):
            if p_val < 0.05:
                cluster_times = grand_related.times[cluster[0]]
                logger.info(f'Cluster {i+1}: {cluster_times[0]*1000:.0f}-{cluster_times[-1]*1000:.0f} ms, p={p_val:.4f}')
        
        times = grand_related.times * 1000
        rel_data = grand_related.copy().pick(n400_chs).get_data().mean(axis=0) * 1e6
        unrel_data = grand_unrelated.copy().pick(n400_chs).get_data().mean(axis=0) * 1e6
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(times, rel_data, color='blue', linewidth=2, label='Related')
        ax.plot(times, unrel_data, color='red', linewidth=2, label='Unrelated')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        
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
    else:
        logger.warning('Not enough epochs for cluster permutation test')


if __name__ == '__main__':
    main()
