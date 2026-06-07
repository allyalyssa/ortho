# N400 ERP Analysis: Semantic Relatedness and Orthographic Neighborhood Density

Secondary analysis of the ERP CORE dataset (Kappenman et al., 2021) 
examining N400 amplitude as a function of semantic relatedness and 
orthographic neighborhood density.

## Key Results

- Unrelated targets elicited significantly more negative N400 amplitudes 
  than related targets (beta = -1.175 uV, z = -3.795, p < 0.001)
- Orthographic neighborhood density showed a marginal negative effect on 
  N400 amplitude (beta = -0.228 uV, z = -1.962, p = 0.050; 
  permutation p = 0.061)
- Condition x density interaction was not significant (p = 0.149)

## Dataset

ERP CORE N400 paradigm (Kappenman et al., 2021). 40 participants, 
27 retained after artifact rejection (mean 65.1 trials per subject).
Raw data: https://osf.io/thsqg/

Citation: Kappenman, E. S., Farrens, J. L., Zhang, W., Stewart, A. X., 
& Luck, S. J. (2021). ERP CORE: An open resource for human 
event-related potential research. NeuroImage, 225, 117465.

## Pipeline

1. fetch_data.py — downloads ERP CORE N400 data via hu-neuro-pipeline
2. preprocess_eeg.py — bandpass 0.1-20Hz, epoch -200-800ms, 
   baseline -200-0ms, artifact rejection >100uV
3. analyze_n400.py — N400 amplitude extraction (300-500ms, Cz/CPz/Pz),
   LME model, permutation test
4. plot_erp.py — grand average ERP waveform and density scatter plot
5. orthographic_density.py — neighborhood density via rapidfuzz

## Statistics

Model: N400_Amplitude ~ Condition + density (MixedLM, random intercept)
N = 1,757 trials, 27 subjects
Condition: beta = -1.175, z = -3.795, p < 0.001
Density: beta = -0.228, z = -1.962, p = 0.050 (permutation p = 0.061)

## Requirements

pip install mne hu-neuro-pipeline statsmodels rapidfuzz pandas numpy matplotlib tqdm
