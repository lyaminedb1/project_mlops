"""
register_models.py — Register all 8 waste-detection models in MLflow.

Run once after `docker compose up -d mlflow`:
    python scripts/register_models.py

Uses only the Model Registry API (no artifact uploads) so it works regardless
of the artifact store configuration. Actual weights are loaded by the API at
inference time (Chapter 2).
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

MODELS = [
    {
        "name": "waste-detector-yolov8",
        "display_name": "YOLOv8",
        "architecture": "YOLOv8n",
        "framework": "ultralytics",
        "description": "Standard YOLOv8 nano object detection model.",
    },
    {
        "name": "waste-detector-yolo26",
        "display_name": "YOLO26",
        "architecture": "YOLOv8n-v26",
        "framework": "ultralytics",
        "description": "Advanced YOLO variant (v26 architecture).",
    },
    {
        "name": "waste-detector-rtdetr",
        "display_name": "RT-DETR",
        "architecture": "RT-DETR-S",
        "framework": "ultralytics",
        "description": "Real-Time DEtection TRansformer, small variant.",
    },
    {
        "name": "waste-detector-rtdetrv2",
        "display_name": "RT-DETRv2",
        "architecture": "RT-DETRv2-N",
        "framework": "transformers",
        "description": "RT-DETR v2 nano via Hugging Face Transformers.",
    },
    {
        "name": "waste-detector-rfdetr",
        "display_name": "RF-DETR",
        "architecture": "RF-DETR-N",
        "framework": "rfdetr",
        "description": "Robustness-focused DETR from Roboflow.",
    },
    {
        "name": "waste-detector-dfine",
        "display_name": "D-FINE",
        "architecture": "D-FINE-N",
        "framework": "transformers",
        "description": "Fine box regression model, nano variant via Hugging Face.",
    },
    {
        "name": "waste-detector-deim-dfine",
        "display_name": "DEIM-DFINE",
        "architecture": "DEIM-D-FINE-L",
        "framework": "deim",
        "description": "D-FINE trained with DEIM training method, large variant.",
    },
    {
        "name": "waste-detector-fusion-model",
        "display_name": "Fusion",
        "architecture": "YOLOv8n + RT-DETR",
        "framework": "ultralytics",
        "description": "Hybrid model combining YOLOv8 backbone with RT-DETR head.",
    },
]


def register_all() -> None:
    client = MlflowClient()

    for info in MODELS:
        name = info["name"]

        # 1. Create the registered model entry (idempotent)
        try:
            client.create_registered_model(
                name=name,
                tags={
                    "architecture": info["architecture"],
                    "framework": info["framework"],
                    "display_name": info["display_name"],
                },
                description=info["description"],
            )
            print(f"[+] Created  : {name}")
        except mlflow.exceptions.MlflowException as exc:
            if "already exists" in str(exc).lower():
                print(f"[~] Exists   : {name}")
            else:
                raise

        # 2. Open a run to attach metadata, then register a version from it.
        #    No model artifact is uploaded — avoids all artifact-store path issues.
        with mlflow.start_run(run_name=f"register-{name}") as run:
            mlflow.log_params({
                "architecture": info["architecture"],
                "framework": info["framework"],
                "display_name": info["display_name"],
            })
            run_id = run.info.run_id

        client.create_model_version(
            name=name,
            source=f"runs:/{run_id}",
            run_id=run_id,
            description=info["description"],
        )
        print(f"    version 1 registered (run {run_id[:8]}…)\n")

    print("All 8 models registered.")
    print(f"Registry UI → {MLFLOW_URI}/#/models")


if __name__ == "__main__":
    register_all()
