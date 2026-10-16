"""Environment-based configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    app_name: str = "Customer Churn Intelligence"
    environment: str = "development"
    database_url: str = ""
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    model_path: str = ""
    metadata_path: str = ""
    segments_path: str = ""
    data_csv_path: str = ""
    max_upload_mb: int = 10
    log_level: str = "INFO"
    auto_import: bool = True

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return "sqlite:///" + (ROOT / "data" / "app.db").resolve().as_posix()

    def resolved_model_path(self) -> Path:
        return Path(self.model_path) if self.model_path else ROOT / "ml" / "artifacts" / "model.joblib"

    def resolved_metadata_path(self) -> Path:
        return Path(self.metadata_path) if self.metadata_path else ROOT / "ml" / "artifacts" / "metadata.json"

    def resolved_segments_path(self) -> Path:
        return Path(self.segments_path) if self.segments_path else ROOT / "ml" / "artifacts" / "segments.joblib"

    def resolved_csv_path(self) -> Path:
        return Path(self.data_csv_path) if self.data_csv_path else ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
