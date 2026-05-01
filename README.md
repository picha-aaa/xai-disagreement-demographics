# The Disagreement Problem in XAI: A Demographic Subgroup Analysis

Course project (ECE2806 (2028) Foundations of AI, Georgia Tech) extending Krishna et al. (2022) to test whether SHAP, LIME, and Grad-CAM disagree more for certain demographic subgroups across tabular and image data.

## Setup

Python 3.10.

pip install -r requirements.txt

## Datasets

|Dataset|Used in|Source|
|---|---|---|
|COMPAS|Experiment 1|https://github.com/propublica/compas-analysis|
|Fitzpatrick17k|Experiment 2|https://github.com/mattgroh/fitzpatrick17k|

## Reproducing results
**Experiment 1** trains logistic regression and a 3-layer neural network (50–100–50, ReLU) on COMPAS, generates SHAP and LIME explanations on the test set, then computes feature, rank, sign, and signed rank agreement metrics split by race and gender.

**Experiment 2** fine-tunes ResNet-18 on Fitzpatrick17k for 3-class skin lesion classification, generates Grad-CAM, LIME, and KernelSHAP attributions, and computes the same agreement metrics split by Fitzpatrick type 1–2 vs. 5–6.

## References
- Krishna et al. (2022). The Disagreement Problem in Explainable Machine Learning. arXiv:2202.01602
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS
- Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD
- Selvaraju et al. (2017). Grad-CAM. ICCV
- Groh et al. (2021). Fitzpatrick 17k Dataset. CVPR Workshops
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR
