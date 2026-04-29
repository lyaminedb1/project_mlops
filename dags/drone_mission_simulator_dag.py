import subprocess
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

SCRIPT_PATH = "/workspace/generate_patrol_db.py"
WORKSPACE_DIR = "/workspace"


def run_patrol_simulation(**context):
    result = subprocess.run(
        ["python", SCRIPT_PATH],
        cwd=WORKSPACE_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)


with DAG(
    dag_id="drone_mission_simulator_dag",
    schedule="*/5 * * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["waste-detection", "simulation"],
) as dag:

    simulate_mission = PythonOperator(
        task_id="simulate_drone_mission",
        python_callable=run_patrol_simulation,
    )

    trigger_sync = TriggerDagRunOperator(
        task_id="trigger_patrol_sync",
        trigger_dag_id="drone_patrol_sync_dag",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    simulate_mission >> trigger_sync
