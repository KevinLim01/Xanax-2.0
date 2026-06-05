from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.WARNING),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("stock_signal_app")

logger = setup_logging()

def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)

def from_json(text: str | None, default: Any = None) -> Any:
    if not text:
        return default
    return json.loads(text)
