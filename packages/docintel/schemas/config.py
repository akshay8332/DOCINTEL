"""Settings and training config (env + YAML via Pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

FUNSD_LABEL_LIST: list[str] = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]


class TrainingConfig(BaseModel):
    model_name: str = "microsoft/layoutlmv3-base"
    fallback_model_name: str = "microsoft/layoutlmv2-base-uncased"
    prefer_v3: bool = True
    data_dir: Path = Path("./data/funsd")
    output_dir: Path = Path("./artifacts/checkpoints/funsd-layoutlm")
    max_seq_length: int = 512
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_train_epochs: float = 5.0
    max_steps: int | None = None
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    seed: int = 42
    fp16: bool = False
    dataloader_num_workers: int = 0
    label_list: list[str] = Field(default_factory=lambda: list(FUNSD_LABEL_LIST))

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCINTEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    artifacts_dir: Path = Path("./artifacts")
    funsd_dir: Path = Path("./data/funsd")
    model_name: str = "microsoft/layoutlmv3-base"
    checkpoint_dir: Path = Path("./artifacts/checkpoints")
    device: str = "auto"
    batch_size: int = 2
    grad_accum: int = 4
    num_epochs: float = 5.0
    learning_rate: float = 5e-5
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    db_url: str = "sqlite:///./artifacts/docintel.db"
    confidence_threshold: float = 0.75
    storage_root: Path = Path("./data/uploads")
    invoice_db_url: str = "sqlite:///./data/app/docintel.db"
    inference_mode: str = "auto"
    frozen_checkpoint: str = "epoch-3"
    max_upload_bytes: int = 20 * 1024 * 1024
    process_inline: bool = False
