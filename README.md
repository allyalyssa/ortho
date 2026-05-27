# N400 ERP Analysis: Semantic Relatedness and Orthographic Density

Secondary analysis of the ERP CORE dataset (Kappenman et al., 2021) 
examining N400 amplitude as a function of semantic relatedness in a 
word-pair judgment task.

## Result

Unrelated target words elicited significantly more negative N400 
amplitudes than related targets (β = -1.114 µV, z = -3.613, p < 0.001) 
across 27 subjects and 1,757 trials. This replicates the canonical N400 
effect and establishes a baseline for subsequent analysis of orthographic 
neighborhood density as a continuous predictor.

## Dataset

ERP CORE N400 paradigm (Kappenman et al., 2021). 40 participants 
completed a word-pair judgment task. Target words were semantically 
related (codes 211, 212) or unrelated (codes 221, 222) to a preceding 
prime. Raw data available at https://osf.io/thsqg/

Citation: Kappenman, E. S., Farrens, J. L., Zhang, W., Stewart, A. X., 
& Luck, S. J. (2021). ERP CORE: An open resource for human 
event-related potential research. NeuroImage, 225, 117465.

## Pipeline

1. fetch_data.py — downloads raw EEG from OSF via hu-neuro-pipeline
2. preprocess_eeg.py — bandpass filter 0.1–20Hz, epoch −200–800ms, 
   baseline −200–0ms, artifact rejection >100µV
3. analyze_n400.py — extracts N400 amplitude (300–500ms, Cz/CPz/Pz), 
   fits linear mixed-effects model
4. plot_erp.py — generates grand average ERP waveform and topographic 
   difference map
5. orthographic_density.py — computes orthographic neighborhood density 
   for stimulus words using rapidfuzz

## Statistics

Linear mixed-effects model (statsmodels MixedLM):
- Fixed effect: Condition (related vs. unrelated)
- Random effect: Subject intercept
- N = 1,757 trials, 27 subjects
- Result: β = -1.114, z = -3.613, p < 0.001

## Next Steps

Adding orthographic neighborhood density as a continuous predictor 
to test whether lexical competition modulates N400 amplitude 
independent of semantic relatedness.

## Requirements

pip install mne hu-neuro-pipeline statsmodels rapidfuzz pandas numpy matplotlib
