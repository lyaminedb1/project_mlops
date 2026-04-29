import sqlite3
import os
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

DRONE_DB_PATH = "/workspace/drone_patrol.db"
APP_DB_PATH = "/data/app_detections.db"
CONFIDENCE_THRESHOLD = 0.65


def extract(**context):
    if not os.path.exists(DRONE_DB_PATH):
        print(f"[extract] {DRONE_DB_PATH} not found — skipping.")
        context["ti"].xcom_push(key="raw_rows", value=[])
        return

    with sqlite3.connect(DRONE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT id, drone_id, timestamp, latitude, longitude, confiance "
            "FROM drone_detections WHERE processed = 0"
        )
        rows = [dict(row) for row in cursor.fetchall()]

    print(f"[extract] Found {len(rows)} unprocessed rows.")
    context["ti"].xcom_push(key="raw_rows", value=rows)


def transform(**context):
    raw_rows = context["ti"].xcom_pull(task_ids="extract", key="raw_rows") or []

    if not raw_rows:
        print("[transform] No rows to transform.")
        context["ti"].xcom_push(key="filtered_rows", value=[])
        return

    filtered = [r for r in raw_rows if r["confiance"] >= CONFIDENCE_THRESHOLD]
    print(
        f"[transform] {len(raw_rows)} rows in → "
        f"{len(filtered)} rows pass threshold {CONFIDENCE_THRESHOLD}."
    )
    context["ti"].xcom_push(key="filtered_rows", value=filtered)


def load(**context):
    filtered_rows = context["ti"].xcom_pull(task_ids="transform", key="filtered_rows") or []

    if not filtered_rows:
        print("[load] No rows to load — nothing to do.")
        return

    os.makedirs(os.path.dirname(APP_DB_PATH), exist_ok=True)
    with sqlite3.connect(APP_DB_PATH) as app_conn:
        app_conn.execute("""
            CREATE TABLE IF NOT EXISTS app_detections (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT,
                latitude   REAL,
                longitude  REAL,
                confiance  REAL,
                model_name TEXT,
                source     TEXT,
                drone_id   TEXT
            )
        """)
        app_conn.executemany(
            """
            INSERT INTO app_detections
                (timestamp, latitude, longitude, confiance, model_name, source, drone_id)
            VALUES (:timestamp, :latitude, :longitude, :confiance, NULL, 'drone_patrol', :drone_id)
            """,
            filtered_rows,
        )
        app_conn.commit()
    print(f"[load] Inserted {len(filtered_rows)} rows into app_detections.")

    ids = [r["id"] for r in filtered_rows]
    placeholders = ",".join("?" * len(ids))
    with sqlite3.connect(DRONE_DB_PATH) as drone_conn:
        drone_conn.execute(
            f"UPDATE drone_detections SET processed = 1 WHERE id IN ({placeholders})",
            ids,
        )
        drone_conn.commit()
    print(f"[load] Marked {len(ids)} source rows as processed=1 in drone_patrol.db.")


with DAG(
    dag_id="drone_patrol_sync_dag",
    schedule="*/10 * * * *",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["waste-detection", "etl"],
) as dag:

    t_extract = PythonOperator(
        task_id="extract",
        python_callable=extract,
    )

    t_transform = PythonOperator(
        task_id="transform",
        python_callable=transform,
    )

    t_load = PythonOperator(
        task_id="load",
        python_callable=load,
    )

    t_extract >> t_transform >> t_load
