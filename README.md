# SubLign code

This repo contains experiments for "Clustering Interval-Censored Data for Disease Phenotyping" (AAAI 2022). The code contains scripts to generate synthetic code for experiment as well as two clinical datasets: 1) Parkinson's disease from the publically-available [Parkinson's Progression Marker Initiative](https://www.ppmi-info.org/access-data-specimens/download-data) and a heart failure collected from Beth Israel Deaconness Medical Center.

Code is released for the purposes of transparency and replication. It has not been extensively cleaned --- nor was it designed to be run directly from this repo. Certain terms have been redacted for security concerns. If there are any questions, please contact Irene Chen at iychen@csail.mit.edu.

Code is written for Python 3.7. 

## Main Paper

### Figure 1: Illustrative plots and graphic model

Figure 1 comprises of three drawings in Powerpoint, which are not algorithmically generated.

### Figure 2: Sigmoid synthetic results and PD results

Figure 2a is generated in two steps. For all commands, `data_num` corresponds to the synthetic data setting. For example, sigmoid data is `data_num=1` whereas the quadratic data settings are integers larger than 1. 

First, best hyperparameters for SubLign and associated performance are found according to `python model/hpsearch.py --data_num 1 --epochs 1000`.

Next, baselines are computed according to corresponding scripts in `baselines`. For example, for KMeans+Loss: `python kmeans.py --data_num 1 --epochs 1000 --trials 5`. Similar format follows for the other baselines, with the exception of PARAFAC2 which was implemented in `parafac.m` (see file for execution instructions).

Figure 2b is generated similarly with the dataset denoted with a `--ppmi` tag: `python model/hpsearch.py --ppmi --epochs 1000`.

### Figure 3: SubLign sigmoid subtypes plotted

Figure 3 is generated in `model/CHF_Experiment.ipynb`

### Table 1: Model Misspecification

Model misspecfication experiments can be found in `model/misspecification.py`. For example: `python misspecification.py --increasing`

### Table 2-4: Missingness results

Table 2-4 are generated similar to Figure 2a with a different data_num. For data with 50% missing, use `data_num=11`. For data with 25% missing, use `data_num=12`. For data with 0% missing, use `data_num=13`.

### Table 5: Quadratic experiment setups

Table 5 is manually created and does not need code to produce.

### Figures 4-9: Quadratic experiments

Similar to the sigmoid experiments, we find best hyperparameters for SubLign and associated performance: `python model/hpsearch.py --data_num 3 --epochs 1000`. The data number is an integer from 3-8 inclusive.

Baselines are computed according to corresponding scripts in `baselines`. For example, for KMeans+Loss: `python kmeans.py --data_num 3 --epochs 1000 --trials 5`.

### Table 6 and 10b: PD subtypes

For PD clinical subtypes, we find the best hyperparameters with `python cross_validation/hpsearch.py --ppmi --epochs 1000` and then compute the corresponding subtypes with `model/ClinicalSubtypes.ipynb`.

### Figure 10: HF KMeans+Loss subtypes

We compute the KMeans+Loss subtypes for the HF dataset in `model/ClinicalSubtypes.ipynb`.

### Figure 11: HF subtypes

For HF clinical subtypes, we find the best hyperparameters with `python cross_validation/hpsearch.py --chf --epochs 1000` and then compute the corresponding subtypes with `model/ClinicalSubtypes.ipynb`.

### HF semi-synthetic experiment

For the HF semi-synthetic experiment, run `python model/run_chf_experiment.py --thresh 0.25 --chf`.
