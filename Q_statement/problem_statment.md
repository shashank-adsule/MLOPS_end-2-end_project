Automated Surface Defect Detection in Manufacturing using Anomaly Detection & MLOps

Problem Statement:
Manufacturing industries face significant quality control challenges, where undetected surface defects lead to product recalls, safety risks, and financial losses. Manual visual inspection is slow, inconsistent, and unscalable. There is a clear need for an automated, reliable, and continuously improving AI system that can detect defects in real time.

Application Domain:
Industrial quality control / Computer Vision

Objective:
The objective is to build an unsupervised anomaly detection system trained exclusively on defect-free product images, capable of identifying and localizing surface defects at inference time. The project will use the publicly available MVTec Anomaly Detection (MVTec AD) dataset, which contains 15 product categories with pixel-level defect annotations.

The core model will be based on the PatchCore algorithm, which constructs a memory bank of normal patch embeddings from a pretrained CNN backbone and flags deviations at inference as anomalies.

Expected Outcome:
A fully functional, containerized AI application that:
- Detects and localizes surface defects with measurable AUROC and F1-score
- Serves predictions via a REST API with inference latency under 200ms
- Monitors for data drift and model performance decay in production
- Automatically triggers model retraining when performance degrades

MLOps Practices:
The project will strictly follow the provided MLOps guidelines across the full AI product lifecycle — including DVC for data versioning, MLflow for experiment tracking and model registry, Docker Compose for a multi-container deployment (API, model server, monitoring), Prometheus and Grafana for alerting, and GitHub Actions for CI/CD pipelines with rollback support. All experiments will be reproducible via Git commit hash and MLflow run ID.



Kaggle link: https://www.kaggle.com/datasets/ipythonx/mvtec-ad