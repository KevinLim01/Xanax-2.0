from __future__ import annotations

import json
from typing import Any

from app.config import settings

try:
    from google import genai
except Exception:
    genai = None


class GeminiJSONClient:
    def __init__(self) -> None:
        self.enabled = bool(
            settings.use_gemini_text_agents
            and settings.gemini_api_key
            and genai is not None
        )
        self._client = None

        if self.enabled:
            self._client = genai.Client(
                api_key=settings.gemini_api_key,
                http_options={"api_version": settings.gemini_api_version},
            )

    def classify(self, system: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled or self._client is None:
            return None

        prompt = (
            system.strip()
            + "\n\nReturn only JSON with double-quoted keys."
            + "\nPayload:\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        try:
            response = self._client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            text = (getattr(response, "text", "") or "").strip()

            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:].strip()

            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                return None

            return json.loads(text[start:end + 1])
        except Exception:
            return None