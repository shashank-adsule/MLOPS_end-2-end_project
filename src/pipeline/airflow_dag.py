import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# ---------------------------------------------------------------------------
# Import project modules (resolved relative to PYTHONPATH in Docker container)
# ---------------------------------------------------------------------------
from src.pipeline.ingest import validate_raw_data
from src.pipeline.preprocess import preprocess_all_categories
from src.pipeline.feature_engineering import compute_baseline_statistics

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAG-level defaults
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "data/raw/mvtec_ad")
PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "data/processed")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "256"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "224"))

# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="mvtec_ad_data_pipeline",
    default_args=DEFAULT_ARGS,
    description="Ingest, validate, preprocess and version MVTec AD dataset",
    schedule_interval=None,          # triggered manually or by CI
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "data-engineering", "mvtec"],
    doc_md=__doc__,
) as dag:

    # ------------------------------------------------------------------
    # Task 1: Validate raw data
    # ------------------------------------------------------------------
    def _validate(**context):
        log.info("Starting raw data validation for %d categories", len(CATEGORIES))
        results = validate_raw_data(
            raw_dir=RAW_DATA_DIR,
            categories=CATEGORIES,
        )
        failed = [cat for cat, ok in results.items() if not ok]
        if failed:
            raise ValueError(f"Validation failed for categories: {failed}")
        log.info("All %d categories passed validation", len(CATEGORIES))
        # Push results to XCom for downstream tasks
        context["ti"].xcom_push(key="validated_categories", value=list(results.keys()))

    validate_task = PythonOperator(
        task_id="validate_raw_data",
        python_callable=_validate,
    )

    # ------------------------------------------------------------------
    # Task 2: Preprocess images
    # ------------------------------------------------------------------
    def _preprocess(**context):
        validated = context["ti"].xcom_pull(
            task_ids="validate_raw_data", key="validated_categories"
        )
        log.info("Preprocessing %d categories", len(validated))
        stats = preprocess_all_categories(
            raw_dir=RAW_DATA_DIR,
            processed_dir=PROCESSED_DATA_DIR,
            categories=validated,
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
        )
        log.info("Preprocessing complete. Stats: %s", stats)
        context["ti"].xcom_push(key="preprocess_stats", value=stats)

    preprocess_task = PythonOperator(
        task_id="preprocess_images",
        python_callable=_preprocess,
    )

    # ------------------------------------------------------------------
    # Task 3: Compute drift baselines
    # ------------------------------------------------------------------
    def _compute_baselines(**context):
        validated = context["ti"].xcom_pull(
            task_ids="validate_raw_data", key="validated_categories"
        )
        log.info("Computing baseline statistics for drift detection")
        for category in validated:
            compute_baseline_statistics(
                processed_dir=PROCESSED_DATA_DIR,
                category=category,
            )
            log.info("Baseline computed for category: %s", category)

    baseline_task = PythonOperator(
        task_id="compute_drift_baselines",
        python_callable=_compute_baselines,
    )

    # ------------------------------------------------------------------
    # Task 4: DVC add + commit processed data
    # ------------------------------------------------------------------
    dvc_task = BashOperator(
        task_id="trigger_dvc_pipeline",
        bash_command="""
            set -e
            echo "Running DVC pipeline..."
            dvc repro --no-commit
            dvc add data/processed
            git add data/processed.dvc
            git commit -m "ci: update processed data [airflow run {{ run_id }}]" || true
            echo "DVC pipeline complete."
        """,
    )

    # ------------------------------------------------------------------
    # Task dependency chain
    # ------------------------------------------------------------------
    validate_task >> preprocess_task >> baseline_task >> dvc_task
s