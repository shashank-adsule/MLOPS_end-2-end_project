"""
Airflow DAG: App Down Response
-------------------------------
Triggered manually (or via API) when the APP DOWN alert is fired
from the Streamlit dashboard. Simulates an incident response workflow:

  1. log_alert_received   - logs the incident
  2. check_services       - checks which services are up
  3. notify_team          - logs notification (extend with email/slack)
  4. attempt_recovery     - placeholder for restart logic
  5. log_resolution       - marks incident resolved
"""

from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ── Task functions ────────────────────────────────────────────────────────────

def log_alert_received(**context):
    triggered_at = context["execution_date"]
    log.warning("=" * 60)
    log.warning("🚨 APP DOWN ALERT RECEIVED")
    log.warning(f"   Triggered at : {triggered_at}")
    log.warning(f"   Run ID       : {context['run_id']}")
    log.warning("=" * 60)
    context["ti"].xcom_push(key="alert_time", value=str(triggered_at))


def check_services(**context):
    import urllib.request
    services = {
        "MLflow":       "http://mlflow:5000/",
        "Prometheus":   "http://prometheus:9090/-/healthy",
        "Alertmanager": "http://alertmanager:9093/-/healthy",
        "Grafana":      "http://grafana:3000/api/health",
    }
    status = {}
    for name, url in services.items():
        try:
            req = urllib.request.urlopen(url, timeout=5)
            status[name] = "UP" if req.status == 200 else f"HTTP {req.status}"
        except Exception as e:
            status[name] = f"DOWN ({e})"
        log.info(f"  {name}: {status[name]}")

    context["ti"].xcom_push(key="service_status", value=status)
    down = [k for k, v in status.items() if "DOWN" in v]
    if down:
        log.warning(f"Services DOWN: {down}")
    else:
        log.info("All monitored services are UP")


def notify_team(**context):
    status = context["ti"].xcom_pull(task_ids="check_services", key="service_status") or {}
    alert_time = context["ti"].xcom_pull(task_ids="log_alert_received", key="alert_time")

    message = f"""
    ==========================================
    🚨 INCIDENT NOTIFICATION
    ==========================================
    Alert    : APP DOWN (manually triggered)
    Time     : {alert_time}
    Run ID   : {context['run_id']}

    Service Health:
    {chr(10).join(f'  {k}: {v}' for k, v in status.items())}

    Action   : Manual intervention required.
    ==========================================
    """
    log.warning(message)
    # ── Extend here: send email / Slack / PagerDuty ──
    # Example Slack webhook:
    # import urllib.request, json
    # payload = json.dumps({"text": message}).encode()
    # urllib.request.urlopen("https://hooks.slack.com/your-webhook", payload)


def attempt_recovery(**context):
    log.info("Attempting automated recovery checks...")
    # ── Extend here with actual restart logic ──
    # e.g. restart a crashed container via Docker API
    log.info("Recovery step complete — no automated action taken.")
    log.info("Operator should manually clear the APP DOWN flag in Streamlit.")


def log_resolution(**context):
    log.info("=" * 60)
    log.info("✅ INCIDENT RESPONSE WORKFLOW COMPLETE")
    log.info("   Next step: operator clears APP DOWN in Streamlit UI")
    log.info("   This will set app_down_manual=0 and resolve the alert")
    log.info("=" * 60)


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="app_down_response",
    default_args=DEFAULT_ARGS,
    description="Incident response workflow triggered by APP DOWN alert",
    schedule_interval=None,      # manual trigger only
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops", "incident-response", "alerting"],
) as dag:

    t1 = PythonOperator(task_id="log_alert_received",  python_callable=log_alert_received)
    t2 = PythonOperator(task_id="check_services",      python_callable=check_services)
    t3 = PythonOperator(task_id="notify_team",         python_callable=notify_team)
    t4 = PythonOperator(task_id="attempt_recovery",    python_callable=attempt_recovery)
    t5 = PythonOperator(task_id="log_resolution",      python_callable=log_resolution)

    t1 >> t2 >> t3 >> t4 >> t5
