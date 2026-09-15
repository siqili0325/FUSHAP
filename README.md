# FUSHAP: Fusion Shapley Attribution from Partially-observed Data

[![arXiv](https://img.shields.io/badge/arXiv-2609.14902-b31b1b.svg)](https://arxiv.org/abs/2609.14902)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)

This repository contains the official Python implementation for **FUSHAP** (Fusion Shapley Attribution from Partially-observed data), a method for estimating Shapley feature attributions from multi-site data with blockwise-missing features, without resorting to imputation.

---

## 📌 Overview

Multi-site studies in biomedicine, environmental monitoring, and social science routinely integrate data from institutions that record different features under different protocols, producing systematic blockwise missingness across sources. Existing Shapley value estimators assume a single, fully observed reference sample and break down under this heterogeneous feature coverage. The standard remedy of imputing missing features before computing Shapley values introduces systematic, coalition-dependent bias into the resulting attributions.

**FUSHAP** avoids imputation entirely. It derives the influence function of the constrained weighted-least-squares Shapley estimator, constructs site-specific control variates from partially observed auxiliary sites, and fuses them directly in attribution space. A permutation-based screening step detects and excludes incompatible sources, and data-adaptive calibration ensures each site's contribution is proportional to its informativeness.

---

## 🌟 Why FUSHAP?

* **No Imputation Needed:** Reduces variance by leveraging auxiliary sites without imputing their missing features, avoiding imputation-induced attribution bias.
* **Influence-Space Fusion:** Compresses perturbations across $2^p$ coalition values into a $p$-dimensional influence representation, enabling data fusion directly in attribution space.
* **Adaptive Screening:** A permutation-based test detects and excludes sites whose data distributions are incompatible with the target population.
* **Variance-Minimizing Calibration:** Optimally weights each site's contribution via a closed-form quadratic program.

---

## 🚀 Quick Start

### Installation

```bash
pip install numpy scipy scikit-learn joblib
```

### Demo

```bash
python demo.py --n_reps 20 --n_jobs 6
```

This runs a simulation demo comparing FUSHAP against single-site, imputation, and oracle baselines. Expected output:

```
  Method                    Mean MSE
  (F) Oracle                0.14
  FUSHAP                    0.25
  (A) LC-only               0.76
  (B) LC+IPW                0.43
  (C) Impute-mean           1.32

  FUSHAP improvement over LC-only: 3.0x
  FUSHAP improvement over Impute:  5.2x
```

---

## ✍️ Citation

If you find our code or paper helpful, please consider citing:

```bibtex
@article{li2026fushap,
  title={Shapley Value Estimation for Multi-Site Data with Blockwise-Missing Features},
  author={Li, Siqi and Fan, Wangxuan and Li, Yiming and Zhou, Doudou and Dai, Zhongxiang and Liu, Molei},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 👥 Contact

Siqi Li — [siqili@u.duke.nus.edu](mailto:siqili@u.duke.nus.edu)
