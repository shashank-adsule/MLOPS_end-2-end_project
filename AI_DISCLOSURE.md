# AI Disclosure Appendix — MLOps End-to-End Project

**Student:** DA25M005 (Shashank Adsule)   
**Repository:** https://github.com/shashank-adsule/MLOPS_end-2-end_project
  
---

## Declaration

This document describes how generative AI tools were used during the development of this MLOps end-to-end project. AI was primarily used as a **debugging assistant and conceptual reference** for understanding Docker networking behaviour, Prometheus metric architecture, MLflow dual-logging patterns, and Streamlit's thread isolation model.

All system design choices, alert threshold justification, pipeline architecture decisions, debugging of container failures, and implementation logic were carried out independently through hands-on experimentation on a local Windows 10 + CUDA environment.

---

## 1. AI Tools Used

| Tool | Model | Purpose |
|---|---|---|
| Claude (Anthropic) | Sonnet 4.x | Debugging Docker compose issues, understanding Prometheus scrape architecture, resolving Streamlit thread isolation problems, and MLflow dual-logging design |

---

## 2. Component-wise Disclosure

### 2.1 Docker Stack Debugging

AI assistance was used to understand the root cause of several container startup failures encountered during development.

**MLflow container fix:** AI helped identify that the original `python:3.10-slim` base image caused gunicorn to bind to `127.0.0.1` (unreachable from the host). The fix — switching to the official `ghcr.io/mlflow/mlflow:v2.12.2` image — was independently verified by testing connectivity from the host after the change.

**Airflow + SQLite crash:** AI was used to understand why SQLite fails under concurrent access from Airflow's scheduler and webserver. After understanding the root cause, I independently chose PostgreSQL (`postgres:15-alpine`) as the replacement and designed the startup sequence (`sleep 20` between scheduler and webserver) based on observed timing behaviour in container logs.

**Prometheus scrape target:** AI clarified how Docker's internal DNS works and specifically that `host.docker.internal` is the correct hostname to reach the Windows host from inside a Docker container. I independently verified this worked by querying `localhost:9090/targets` after the fix.

### 2.2 Prometheus Metrics Architecture

AI assistance was used to understand why module-level Python objects are not shared across Streamlit threads.

**Key insight from research:** Each button click or file upload in Streamlit may execute in a separate thread — Counter increments in one thread have no effect on what another thread serves. After understanding this, I independently designed the **file-based metrics state** approach: prediction stats are written to `app_metrics_state.json` in `%TEMP%` on every inference, and the custom `_MetricsHandler` reads this file fresh on every Prometheus scrape.

The decision to use a private `CollectorRegistry` (`_APP_REGISTRY`) to avoid duplicate registration conflicts was also made independently after observing registration errors during development.

AI was used to understand the conceptual difference between Prometheus Counters and Gauges. The decision to use Gauges for all metrics (because values come from file reads rather than in-process increments) was made independently.

### 2.3 MLflow Dual Logging

AI was consulted to understand the MLflow tracking URI override pattern when logging to two different servers simultaneously.

The final dual-logging design in `src/model/train.py` — where DagsHub is the primary and local MLflow is secondary, with graceful fallback if local MLflow is down — was designed and implemented independently. The `dagshub_run_id` cross-referencing tag was added independently as a practical navigability improvement.

### 2.4 Alert Rules Design

AI was used to reference correct **PromQL syntax structure** for rate-based expressions and histogram quantile queries.

Alert threshold values were decided independently after observing actual system behaviour:

- The **200 ms latency SLA** directly matches the `latency_target_ms` in `params.yaml`, which was chosen based on measured GPU inference times across all 15 categories (actual average: 86.6 ms, worst case: 166 ms for wood).
- The **80% defect rate** threshold was chosen after observing typical category defect ratios in the MVTec AD dataset — a rate this high would be statistically anomalous and indicate model or data issues rather than genuine defects.
- The **AppDown immediate threshold** (`for: 0m`) was chosen because it is manually triggered by an operator, so a delay would serve no purpose.

### 2.5 Airflow DAG Design

AI was used to clarify correct Airflow DAG syntax patterns, particularly for `schedule_interval=None` to create a manual-only DAG.

The two-DAG architecture — separating the data pipeline DAG from the incident response DAG — was an independent design decision made to keep operational concerns separate from data engineering concerns. All five tasks in the `app_down_response_dag` and their sequencing were independently designed.

### 2.6 Grafana Dashboard

AI was used to verify correct datasource UID references in JSON dashboard provisioning. The root cause of all 13 panels returning no data (mismatched datasource UIDs `defect-detection-main` vs `prometheus`) was identified independently through Grafana's panel inspector.

Dashboard panel layout, grouping logic, metric selection, and the overall monitoring story (latency SLA → prediction rates → defect ratios → drift → alerts) were designed independently.

### 2.7 APP DOWN Manual Alert

The idea to add a manual alert trigger to the Streamlit sidebar was an independent design decision — not based on AI suggestions. AI was consulted only to verify that `for: 0m` is valid PromQL syntax for an immediate-fire alert.

The complete implementation — file-based state sharing via `%TEMP%\app_down_state.txt`, the custom HTTP metrics handler, and the sidebar button with proper `st.rerun()` state management — was written and debugged independently.

---

## 3. Key Independent Technical Decisions

- **PatchCore over supervised methods** — chosen because MVTec AD training sets contain only normal images; supervised defect classifiers would require labelled defect samples that do not exist in sufficient quantity for all 15 categories.
- **WideResNet50 feature layers `layer2` + `layer3`** — chosen to balance feature resolution (spatial detail) vs semantic abstraction, based on reading the original PatchCore paper.
- **Coreset ratio of 10%** — chosen as a balance between memory bank size and inference speed; a larger coreset improves accuracy marginally but increases nearest-neighbour search time.
- **File-based metrics state** as the thread-safety solution for Prometheus in Streamlit — selected over alternatives (multiprocessing shared memory, Redis) for simplicity and zero additional infrastructure.
- **Dual MLflow logging** — designed so that DagsHub logging (with artifact upload) is never blocked by a local MLflow outage.
- **Postgres for Airflow** — switched from SQLite after observing gunicorn timeout errors caused by SQLite's write lock under concurrent scheduler + webserver access.

---

## 4. AI-Assisted Areas (Summary)

AI contributed primarily in:

- Understanding Docker networking (`host.docker.internal`) and container DNS behaviour
- Clarifying Prometheus metric type semantics (Gauge vs Counter vs Histogram)
- PromQL syntax patterns for rate and quantile expressions
- Streamlit thread isolation behaviour and its implications for shared state
- MLflow tracking URI override patterns for multi-server logging

All final implementations were manually written, tested, and tuned based on observed system behaviour on the local development environment.

---

## 5. Learning Outcomes

Through this project I strengthened practical understanding of:

- End-to-end MLOps pipeline design — from raw data to monitored production inference
- Container networking and the implications of Docker's internal DNS for observability
- Prometheus scrape architecture and the challenges of metric sharing in multi-threaded Python applications
- PatchCore anomaly detection and the trade-offs in coreset memory bank design
- MLflow experiment tracking at scale across 15 trained model categories
- Grafana dashboard design for real-time ML system observability
- Airflow DAG design for both data pipelines and operational incident response

AI assistance helped accelerate **initial conceptual exploration and syntax verification**, while deeper understanding of the system developed through debugging real failures — container crashes, scrape target errors, thread isolation issues, and alert routing problems — all identified and resolved through hands-on experimentation.

---

## Academic Integrity Statement

I confirm that I understand the full monitoring architecture, all PromQL expressions, alert threshold choices, Prometheus metric design, and system behaviour demonstrated in this project. This disclosure accurately reflects how AI tools were used during development.

**Prepared by:** Shashank Adsule  
**Student ID:** DA25M005  
**Date:** April 2026  
