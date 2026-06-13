import logging
import mne
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

RELATED_CODES = {'211', '212'}
UNRELATED_CODES = {'221', '222'}
TARGET_CODES = RELATED_CODES | UNRELATED_CODES
ICA_DIR = Path('data/derivatives/preprocessed_ica')

def extract_trials(density_df: pd.DataFrame) -> pd.DataFrame:
    related_density = density_df[density_df['condition'] == 'related']['density'].values
    unrelated_density = density_df[density_df['condition'] == 'unrelated']['density'].values
    rows = []
    for epo_file in sorted(ICA_DIR.glob('sub-*_N400_ica-epo.fif')):
        sub = epo_file.name.split('_')[0]
        epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)
        if len(epochs) < 30:
            logger.warning(f'Excluding {sub}: only {len(epochs)} epochs')
            continue
        target_id = {k: v for k, v in epochs.event_id.items() if str(k) in TARGET_CODES}
        if not target_id:
            logger.warning(f'{sub}: no target event IDs found')
            continue
        reverse_id = {v: str(k) for k, v in target_id.items()}
        target_mask = np.isin(epochs.events[:, 2], list(target_id.values()))
        target_epochs = epochs[target_mask]
        if len(target_epochs) < 30:
            logger.warning(f'Excluding {sub}: only {len(target_epochs)} target epochs')
            continue
        n400_chs = [c for c in target_epochs.ch_names if c in ['Cz', 'CPz', 'Pz']]
        if not n400_chs:
            n400_chs = target_epochs.ch_names[:3]
        data = target_epochs.copy().crop(tmin=0.3, tmax=0.5).get_data(picks=n400_chs)
        amp = data.mean(axis=(1, 2)) * 1e6
        related_idx = 0
        unrelated_idx = 0
        for i, event in enumerate(target_epochs.events):
            code_str = reverse_id.get(event[2], '')
            if not code_str:
                continue
            condition = 'related' if code_str in RELATED_CODES else 'unrelated'
            if condition == 'related':
                density = related_density[related_idx % len(related_density)]
                related_idx += 1
            else:
                density = unrelated_density[unrelated_idx % len(unrelated_density)]
                unrelated_idx += 1
            rows.append({'Subject': sub, 'N400_Amplitude': amp[i], 'Condition': condition, 'density': density})
        logger.info(f'{sub}: {len(target_epochs)} epochs')
    return pd.DataFrame(rows)

def report_significance(label: str, coef: float, z: float, p: float) -> None:
    sig = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    logger.info(f'{label}: beta = {coef:.3f}, z = {z:.3f}, {sig}')

def run_permutation(df: pd.DataFrame, n_perm: int = 2000) -> float:
    observed = smf.mixedlm('N400_Amplitude ~ Condition + density', df, groups=df['Subject']).fit(disp=False).fe_params['density']
    null_betas = []
    rng = np.random.default_rng(42)
    for _ in tqdm(range(n_perm), desc='Permutation test'):
        df['density_perm'] = rng.permutation(df['density'].values)
        m = smf.mixedlm('N400_Amplitude ~ Condition + density_perm', df, groups=df['Subject']).fit(disp=False)
        null_betas.append(m.fe_params['density_perm'])
    null_betas = np.array(null_betas)
    perm_p = float(np.mean(np.abs(null_betas) >= np.abs(observed)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_betas, bins=50, color='gray', alpha=0.7, label='Null distribution')
    ax.axvline(observed, color='red', linewidth=2, label=f'Observed beta={observed:.3f}')
    ax.set_xlabel('Density coefficient')
    ax.set_ylabel('Count')
    ax.set_title(f'Permutation test: density effect (p={perm_p:.4f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/permutation_density.png', dpi=300)
    return perm_p

if __name__ == '__main__':
    density_df = pd.read_csv('data/stimuli_density.csv')
    df = extract_trials(density_df)
    if df.empty:
        logger.error('No trials extracted. Check ICA epoch files.')
        raise SystemExit(1)
    logger.info(f'Total trials: {len(df)}, Subjects: {df["Subject"].nunique()}')
    logger.info(f'Related mean: {df[df.Condition=="related"]["N400_Amplitude"].mean():.3f} uV')
    logger.info(f'Unrelated mean: {df[df.Condition=="unrelated"]["N400_Amplitude"].mean():.3f} uV')
    logger.info('--- Model 1: Condition + density ---')
    m1 = smf.mixedlm('N400_Amplitude ~ Condition + density', df, groups=df['Subject']).fit()
    for line in str(m1.summary()).split('\n'):
        logger.info(line)
    logger.info('--- Model 2: Condition * density ---')
    m2 = smf.mixedlm('N400_Amplitude ~ Condition * density', df, groups=df['Subject']).fit()
    for line in str(m2.summary()).split('\n'):
        logger.info(line)
    perm_p = run_permutation(df)
    logger.info(f'Permutation p-value: {perm_p:.4f}')
