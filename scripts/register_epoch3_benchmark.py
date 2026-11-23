"""Register the frozen epoch-3 checkpoint in the local MLflow server.

The exact-match counts are the published 100-page audit. This script does not
load the weights, retrain, or rescore the invoices.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

EXPERIMENT = "docile-address-extraction"
MODEL_NAME = "docile-address-epoch3"
FIELD_NAMES = (
    "document_id,date_issue,date_due,vendor_name,customer_billing_name,"
    "vendor_address,customer_billing_address,amount_due,amount_total_gross"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_id(client: MlflowClient) -> str:
    found = client.get_experiment_by_name(EXPERIMENT)
    if found is not None:
        return found.experiment_id
    return client.create_experiment(EXPERIMENT)


def run_with_role(client: MlflowClient, experiment: str, role: str):
    found = client.search_runs(
        [experiment],
        filter_string=f"tags.role = '{role}'",
        max_results=1,
    )
    return found[0] if found else None


def pointer(folder: Path) -> dict:
    weights = folder / "model.pt"
    if not weights.is_file():
        raise SystemExit(f"Missing checkpoint weights: {weights}")
    trainer = json.loads((folder / "trainer_state.json").read_text(encoding="utf-8"))
    return {
        "model_pt_sha256": sha256_file(weights),
        "model_pt_bytes": weights.stat().st_size,
        "best_epoch": trainer.get("best_epoch"),
        "best_val_loss": trainer.get("best_val_loss"),
        "train_pages": trainer.get("train_pages"),
        "learning_rate": trainer.get("learning_rate"),
    }


def log_json(name: str, payload: dict) -> None:
    path = Path("/tmp") / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(path))


def register_benchmark(client: MlflowClient, experiment: str, folder: Path) -> None:
    existing = run_with_role(client, experiment, "benchmark")
    if existing is not None:
        print(f"Benchmark run already recorded: {existing.info.run_id}")
        return
    details = pointer(folder)
    with mlflow.start_run(experiment_id=experiment, run_name="epoch-3-benchmark"):
        mlflow.set_tags({"role": "benchmark", "selected": "true", "stage": "benchmark"})
        mlflow.log_params(
            {
                "model_version": "layoutlmv3-multilabel-epoch3",
                "checkpoint_reference": "docile_multilabel_full_5180/epoch-3",
                "checkpoint_dir": "artifacts/checkpoints/epoch-3",
                "threshold": "0.50",
                "grouping": "line-block-v2",
                "correction_version": "stack-v1:correct+boundary+fallback",
                "ocr_engine": "docTR",
                "architecture": "layoutlmv3-base + 9 sigmoid heads",
                "field_names": FIELD_NAMES,
                "seed": "42",
                "learning_rate": str(details["learning_rate"]),
            }
        )
        mlflow.log_metrics(
            {
                "raw_exact": 61,
                "raw_total": 162,
                "stack_exact": 72,
                "vendor_raw_exact": 33,
                "vendor_raw_total": 92,
                "customer_raw_exact": 28,
                "customer_raw_total": 70,
                "false_positive_words": 171,
                "best_val_loss": float(details["best_val_loss"]),
            }
        )
        log_json("checkpoint_pointer.json", details)
        run = mlflow.active_run()
        run_id = run.info.run_id
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException as exc:
        if "RESOURCE_ALREADY_EXISTS" not in str(exc):
            raise
    version = client.create_model_version(
        MODEL_NAME,
        source=f"runs:/{run_id}",
        run_id=run_id,
        description=(
            "Frozen epoch-3 benchmark. Raw exact 61/162. "
            "Corrected exact 72/162. Threshold 0.50."
        ),
    )
    client.set_registered_model_alias(MODEL_NAME, "benchmark", version.version)
    client.set_model_version_tag(MODEL_NAME, version.version, "stage", "benchmark")
    print(f"Registered {MODEL_NAME} version {version.version} from run {run_id}")


def log_rejected(client: MlflowClient, experiment: str, folder: Path) -> None:
    weights = folder / "model.pt"
    if not weights.is_file():
        print("Party fine-tune checkpoint is not mounted. Skipped the rejected run.")
        return
    existing = run_with_role(client, experiment, "rejected")
    if existing is not None:
        print(f"Rejected run already recorded: {existing.info.run_id}")
        return
    details = pointer(folder)
    with mlflow.start_run(experiment_id=experiment, run_name="party-fine-tune-rejected"):
        mlflow.set_tags({"role": "rejected", "selected": "false", "stage": "rejected"})
        mlflow.log_params(
            {
                "checkpoint_dir": "artifacts/checkpoints/docile_party_experiment",
                "threshold": "0.50",
                "grouping": "line-block-v2",
                "correction_version": "stack-v1:correct+boundary+fallback",
                "conclusion": "did not replace epoch-3",
            }
        )
        mlflow.log_metrics(
            {
                "raw_exact": 59,
                "raw_total": 162,
                "stack_exact": 72,
                "customer_raw_exact": 26,
                "customer_raw_total": 70,
                "best_val_loss": float(details["best_val_loss"]),
            }
        )
        log_json("party_checkpoint_pointer.json", details)
        run_id = mlflow.active_run().info.run_id
    print(f"Recorded rejected party fine-tune run {run_id}")


def main() -> None:
    tracking = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking)
    client = MlflowClient(tracking_uri=tracking)
    experiment = experiment_id(client)
    register_benchmark(
        client,
        experiment,
        Path(os.environ.get("DOCINTEL_CHECKPOINT_PATH", "/checkpoint")),
    )
    log_rejected(
        client,
        experiment,
        Path(os.environ.get("DOCINTEL_PARTY_CHECKPOINT_PATH", "/party-checkpoint")),
    )


if __name__ == "__main__":
    main()
