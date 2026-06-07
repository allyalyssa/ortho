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

if __name__ == '__main__':
    density_df = pd.read_csv('data/stimuli_density.csv')
    related_density = density_df[density_df['condition'] == 'related']['density'].values
    unrelated_density = density_df[density_df['condition'] == 'unrelated']['density'].values

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
            logger.warning(f'{sub}: no target events found')
            continue
        reverse_id = {v: str(k) for k, v in target_id.items()}
        epochs = mne.Epochs(raw, events, event_id=target_id,
                            tmin=-0.2, tmax=0.8,
                            baseline=(-0.2, 0),
                            reject=dict(eeg=100e-6),
                            preload=True, verbose=False)
        if len(epochs) < 30:
            logger.warning(f'Excluding {sub}: only {len(epochs)} epochs')
            continue
        n400_chs = [c for c in epochs.ch_names if c in ['Cz', 'CPz', 'Pz']]
        if not n400_chs:
            n400_chs = epochs.ch_names[:3]
        data = epochs.copy().crop(tmin=0.3, tmax=0.5).get_data(picks=n400_chs)
        amp = data.mean(axis=(1, 2)) * 1e6
        related_idx = 0
        unrelated_idx = 0
        for i, event in enumerate(epochs.events):
            code_str = reverse_id[event[2]]
            condition = 'related' if code_str in RELATED_CODES else 'unrelated'
            if condition == 'related':
                density = related_density[related_idx % len(related_density)]
                related_idx += 1
            else:
                density = unrelated_density[unrelated_idx % len(unrelated_density)]
                unrelated_idx += 1
            rows.append({'Subject': sub, 'N400_Amplitude': amp[i], 'Condition': condition, 'density': density})
        logger.info(f'{sub}: {len(epochs)} epochs')

    df = pd.DataFrame(rows)
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

    logger.info('--- Permutation test (n=2000) ---')
    observed = m1.fe_params['density']
    null_betas = []
    rng = np.random.default_rng(42)
    for _ in tqdm(range(2000), desc='Permutation'):
        df['density_perm'] = rng.permutation(df['density'].values)
        m_perm = smf.mixedlm('N400_Amplitude ~ Condition + density_perm', df, groups=df['Subject']).fit(disp=False)
        null_betas.append(m_perm.fe_params['density_perm'])
    null_betas = np.array(null_betas)
    perm_p = float(np.mean(np.abs(null_betas) >= np.abs(observed)))
    logger.info(f'Observed density beta: {observed:.3f}')
    logger.info(f'Permutation p-value: {perm_p:.4f}')
