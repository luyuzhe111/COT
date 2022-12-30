# Predicting Out-of-Distribution Error with Optimal Transport

```run_fot.py``` implements our proposed Feature Optimal Transport algorithm.

```run_projnorm.py``` implements a recent SOTA method ProjNorm.

```run_baselines.py``` implements three other baselines, ConfScore, Entropy, and ATC. 

```notebooks``` folder includes code to gather results and make visualizations. 

Notes:

1) There are no hyperparameters to tune for our method. For ProjNorm, we closely followed the optimal hyperparameters listed in their [ICML 2022]((https://arxiv.org/abs/2202.05834)) paper.  