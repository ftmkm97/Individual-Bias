<div align="center">
  <h1>Individual Fairness</h1>
  <h2>Who Gets to be an Expert? The Hidden Bias in Expert Finding</h2>
</div>


## Introduction
This repository contains the analysis and findings related to the paper, "Who Gets to be an Expert? The Hidden Bias in Expert Finding". This research investigates  individual biases in expert-finding (EF) algorithms in Community Question Answering (CQA) platforms. EF systems are designed to identify users who can provide high-quality answers. However, these systems often exhibit biases that may lead to inequitable recognition of users, impacting the fairness and accuracy of expert recommendations.
The paper defines several metrics to quantify individual bias in expert finding (EF) systems. Here are the key metrics:
- **Individual Bias (IB)**: This metric measures the discrepancy in the distribution of user attributes (like activity score, badge score, and reputation score) between the full set of answerers and those selected by the EF method. 
- **Bias Impact Score (BIS)**: This metric measures the proportion of incorrect predictions where the attribute value of the incorrectly recommended user is higher than that of the correct answerer:
- **Attribute Distribution Score (ADS)**: evaluates how an accepted answerer’s attribute aligns with the distribution of all other answerers using Percentile Rank.

## Contents
- **Data**/: Contains the preprocessed datasets and baseline results used in our experiments.
- **Charts**/: Includes visualizations from the paper.
- **Results**/: Stores the results of bias formula computations and the evaluation metrics of the baseline methods.


## Code Structure
- **Charts.py**: Scripts for generating visualizations used in the paper.
- **Evaluation.py**: Scripts for evaluating EF meyhods.
- **IndividualBias.py**: Scripts for calculating individual bias and the proposed bias impact score (BIS).
- **main.py**: Main script to evaluate EF methods, calculate metrics, and save results.

## Datasets
This study use data from multiple Stack Overflow domains,including 3DPrinting, AI, Bioinformatics, Biology, and History. The Stack Overflow data is publicly available [here](https://archive.org/details/stackexchange). 


## Baseline Models
The paper evaluate various expert finding (EF) methods to analyze their performance regarding individual bias. Here are the baselines used in their experiments:

- [BM25]: A ranking method based on the BM25 score between the target question and previously answered questions.
- [PMEF](https://github.com/pengqy/WWW2022_PMEF): Employs a multi-view attentive matching model with question and view-specific encoders to capture expert-question relationships.
- [BGER](https://github.com/vaibhavkrshn/UMAP-t-BGER): Uses graph diffusion incorporating semantic and temporal data to capture evolving user expertise. 
- [TUEF](https://github.com/maddalena-amendola/TUEF): A topic-based multi-layer graph approach that links user content and social information. We adapt it by treating all answerers as potential experts to standardize evaluation. 
- [DSSM](https://github.com/NTMC-Community/MatchZoo-py.git): A learn-to-rank model based on historical responses, aggregating questions previously answered by each user.
