# breast-cancer-optimism-audit
Code and digitized datasets for optimism ladder audit and digital twin forecasting in breast cancer.
# Optimism Ladder Audit and Digital-Twin Forecasting

This repository contains the reproduction pipeline, code, and digitised datasets for evaluating data-leakage optimism ($L_1$ to $L_5$) and digital-twin forecasts in breast cancer.

## Repository Contents
- `CaseStudy1_MetastasisLadder.ipynb`: Reproduces the 5-rung optimism audit across breast cancer microarray cohorts (DM5 and Bone metastasis).
- `PECT_DigitalTwin_Audit_v3.ipynb`: Implements patient-specific digital twin forecasting, Kalman filtering, and power simulations.
- `pect_lesions_digitised.csv` & `pect_animal_digitised.csv`: Digitised lesion kinetics and preclinical electrochemotherapy datasets.
- `Supplementary_Material.pdf`: Extended statistical derivations, TRIPOD+AI patient flow, and complete performance tables (Tables S1–S14).

## Requirements
- Python >= 3.9
- scikit-learn >= 1.8
- numpy, scipy, pandas, matplotlib
