# breast-cancer-optimism-audit

Code, digitised data and results for the paper *How Much Reported Breast Cancer AI Accuracy Survives Honest Validation? An Optimism-Ladder Audit of Metastasis Classifiers and an Audit-Ready Digital Twin*.

The paper re-evaluates the same models on the same data under five progressively weaker validation schemes, from an unseen cohort or patient (L1) to resubstitution (L5). Case study 1 applies this optimism ladder to six classifier families for five-year distant metastasis and bone relapse in four public microarray cohorts (783 patients). Case study 2 builds an audit-ready digital twin of lesion response to positive electrostatic charge therapy (PECT) from digitised published data (11 lesions, 2 patients) and tests it with the same ladder.

## Repository contents

| File | Contents |
|---|---|
| `CaseStudy1_MetastasisLadder.ipynb` | Case study 1. Downloads GSE2034, GSE7390, GSE11121 and GSE2603 from NCBI GEO, builds the gene-level compendium, the patient flow and the strict L1 train/test matrix (Table II), and runs the optimism ladder (Table III, Fig. 1a), the sensitivity analyses, DeLong comparisons, baselines, calibration, decision curves and signature stability. Writes `patient_flow.csv` and `casestudy1_results.json`. |
| `PECT_DigitalTwin_Audit_v3.ipynb` | Case study 2. Digital twin (empirical-Bayes prior, Kalman update and drift) and its rival models, the nested twin ladder with lesion-bootstrap CIs, per-patient errors, the exhaustive within-patient permutation test (11,660 rearrangements), the power simulation, the illustrative cross-species prior and Fig. 2. Writes the two digitised CSV files and `twin_v3_results.json`. |
| `revision_analyses.py` | Bootstrap CIs for the external calibration slopes (Table IV), the comparison of included and excluded patients, and Fig. 1. Uses the files written by the two notebooks. |
| `pect_lesions_digitised.csv` | Lesion volumes (mm³) for patients 2 and 3, digitised from Zandi et al. 2023, *Front. Med.*, Fig. 3E and 3G. |
| `pect_animal_digitised.csv` | Group-mean tumour volumes from the preclinical PECT mouse study, digitised from Zandi et al. 2021, *Cancer Med.*, Fig. 3D(ii). |
| `patient_flow.csv` | Patient flow per cohort (TRIPOD+AI): arrays, evaluable patients, events and exclusions for each endpoint. |
| `casestudy1_results.json` | All numerical results of case study 1. |
| `twin_v3_results.json` | All numerical results of case study 2. |
| `Supplementary_Material_v4.pdf` | Supplementary Tables S1–S14: patient flow and the included-versus-excluded comparison, complete model settings, definitions of the AUC summaries, per-cohort AUCs, calibration, decision curves, baselines, and all digital-twin results with the forecast-error derivation. |

## Reproducing the results

1. Run `CaseStudy1_MetastasisLadder.ipynb`. It downloads the GEO files into `geo/` and takes about 15 minutes on two CPU cores. Set `FAST = True` in the first cell for a quick check with fewer repeats.
2. Run `PECT_DigitalTwin_Audit_v3.ipynb` (a few minutes).
3. Run `python revision_analyses.py` in the same folder. It writes `calibration_ci.json` and `Fig1_600dpi.png`.

Random seeds are fixed in the code (20260806 for case study 1; a separate seed for each analysis in case study 2), so a rerun gives the released numbers.

## Requirements

- Python 3.9 or later
- scikit-learn 1.8 (the released results were produced with 1.8.0; later versions change the tree-based models slightly)
- numpy, scipy, pandas, matplotlib
- Jupyter, to run the notebooks
