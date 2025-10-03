# Robust Bayesian Inference for Measurement Error Misspecification: The Berkson and Classical Cases
This repository contains all code needed to recreate the results in the paper "Robust Bayesian Inference for Measurement Error Misspecification: The Berkson and Classical Cases" by 
Charita Dellaporta and Theodoros Damoulas. Our method uses [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) which is compatible with GPU usage. We use the R code (`mental_health_study/bcr_sim.R`) 
provided by Harezlak, Ruppert & Wand (2018) [here](https://cran.r-project.org/web/packages/HRW/index.html) to compare against the BCR method and the R code provided by the [SIMEX](https://CRAN.R-project.org/package=simex) package. We further use the California school dataset 
provided by the AER R package [here](https://cran.r-project.org/web/packages/AER/AER.pdf). 

The `requirements.txt` file contains all the requirements. 

## Reproducing experiments
- The `train.py`, `train_tls.py`, `train_classical.py` and `train_cas.py` contain the scripts needed to reproduce the non-linear example (with Berkson error), the linear regression (with the TLS loss),
  the non-linear regression with Classical error and the California School Test experiment respectively.
- The `mental_health_study` folder contains code to reproduce the mental health study experiment (more details are provided within the folder).
- The `gather_results_*` notebooks provide the necessary code to process the outputs of the scripts and plot/print results.

## References
Harezlak, J., Ruppert, D. and Wand, M.P., 2018. Semiparametric regression with R (Vol. 109). New York: Springer.
