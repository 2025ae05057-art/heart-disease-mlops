# End-to-End MLOps for Heart Disease Classification

**Course:** AIMLCZG523 — MLOps, Assignment 01  
**Student:** [Your name / ID]  
**Date:** [Submission date]

## Abstract
Describe the objective, dataset, models, deployment and measured results in your own words after running the pipeline.

## 1. Problem statement and scope
Explain the binary classification task and clarify that the API is an educational demonstration, not a clinical diagnostic system.

## 2. Dataset acquisition and data quality
Record the source URL, row/column counts, target conversion, missing values, data split and any limitations. Include the data summary and missing-value table.

## 3. Exploratory data analysis
Insert `class_balance.png`, `feature_histograms.png` and `correlation_heatmap.png`. Explain actual patterns you observe; do not invent conclusions.

## 4. Feature engineering and preprocessing
Describe numeric/categorical feature groups, median/mode imputation, standardization and one-hot encoding. Explain why fitting preprocessing inside a pipeline avoids leakage.

## 5. Modelling, validation and model selection
Document Logistic Regression and Random Forest search spaces, stratified five-fold CV, random seed and selection criterion. Insert measured comparison table from `model_comparison.json`.

## 6. Experiment tracking
Add MLflow UI screenshot. Explain run names, parameters, CV ROC-AUC, held-out metrics, saved model and logged plots.

## 7. Packaging and reproducibility
Describe Python version, requirements, saved pipeline, directory layout and clean setup commands.

## 8. Automated testing and CI/CD
Describe tests and GitHub Actions stages. Insert passing workflow screenshot and uploaded artifact evidence.

## 9. API, Docker and Kubernetes deployment
Include API request/response, Docker build/run screenshots, manifest overview, rollout/pod output and service access method.

## 10. Monitoring, limitations and conclusion
Include request logs, `/metrics` and dashboard screenshot if implemented. State any incomplete items honestly. Discuss small dataset size, data provenance, generalization and lack of clinical validation.

## References
Cite the UCI dataset and any libraries/documentation actually used.
