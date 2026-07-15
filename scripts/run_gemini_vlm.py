#!/usr/bin/env python3
"""List reconstructable foreground objects with the Gemini Interactions API."""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "prompt": {"type": "string"},
                    "fallback_prompt": {"type": "string"},
                    "bbox_xyxy": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "support_parent_id": {"type": "integer", "minimum": -1},
                    "support_relation": {
                        "type": "string",
                        "enum": ["on", "inside", "none"],
                    },
                },
                "required": [
                    "name",
                    "prompt",
                    "fallback_prompt",
                    "bbox_xyxy",
                    "support_parent_id",
                    "support_relation",
                ],
            },
        }
    },
    "required": ["objects"],
}


def load_prompt(value: str) -> str:
    path = Path(value).expanduser()
    return path.read_text(encoding="utf-8").strip() if path.is_file() else value.strip()


def image_block(path: Path) -> dict[str, str]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return {
        "type": "image",
        "mime_type": mime_type,
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
    }


def build_request(image: Path, prompt: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": [{"type": "text", "text": prompt}, image_block(image)],
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": OBJECT_SCHEMA,
        },
        "store": False,
    }


def request_gemini(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or GOOGLE_API_KEY.")
    url = f"{args.api_base.rstrip('/')}/interactions"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini HTTP {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"Gemini request failed: {exc}")
        if attempt < args.retries:
            time.sleep(args.retry_delay * (attempt + 1))
    assert last_error is not None
    raise last_error


def extract_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text") or response.get("outputText")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    for step in reversed(response.get("steps", [])):
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for block in reversed(step.get("content", [])):
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                return str(block["text"]).strip()

    outputs = response.get("output") or response.get("outputs") or []
    for item in reversed(outputs if isinstance(outputs, list) else []):
        if isinstance(item, dict) and item.get("text"):
            return str(item["text"]).strip()

    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if isinstance(part, dict) and part.get("text"):
                return str(part["text"]).strip()
    raise RuntimeError("Gemini response did not contain text output.")


def parse_objects(text: str) -> dict[str, list[dict[str, Any]]]:
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    data = json.loads(text)
    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, list) or not objects:
        raise ValueError("Gemini output must contain a non-empty objects list.")

    normalized = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"Object {index} is not a JSON object.")
        prompt = str(item.get("prompt") or "").strip()
        fallback = str(item.get("fallback_prompt") or item.get("name") or "").strip()
        bbox = item.get("bbox_xyxy")
        parent_id = int(item.get("support_parent_id", -1))
        relation = str(item.get("support_relation") or "none").lower()
        if not prompt or not fallback or not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Object {index} is missing prompt, fallback_prompt, or bbox_xyxy.")
        if relation not in {"on", "inside", "none"}:
            raise ValueError(f"Object {index} has an invalid support_relation.")
        if parent_id >= index or parent_id < -1:
            raise ValueError(
                f"Object {index} support_parent_id must reference an earlier object or -1."
            )
        if (parent_id == -1) != (relation == "none"):
            raise ValueError(
                f"Object {index} support_parent_id and support_relation disagree."
            )
        bbox = [float(value) for value in bbox]
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in bbox):
            raise ValueError(f"Object {index} bbox_xyxy must use normalized coordinates.")
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(f"Object {index} bbox_xyxy is empty or reversed.")
        normalized.append(
            {
                "name": str(item.get("name") or fallback).strip(),
                "prompt": prompt,
                "fallback_prompt": fallback,
                "bbox_xyxy": bbox,
                "support_parent_id": parent_id,
                "support_relation": relation,
            }
        )
    return {"objects": normalized}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True, help="Prompt text or path.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("GEMINI_TEXT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.environ.get("GEMINI_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    objects = parse_objects(extract_output_text(request_gemini(build_request(image, load_prompt(args.prompt), args.model), args)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(objects, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[gemini-vlm] ERROR: {exc}", file=sys.stderr)
        raise
