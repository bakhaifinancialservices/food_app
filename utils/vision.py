# utils/vision.py
# ─────────────────────────────────────────────────────────────
# Food detection from images using Groq Llama 4 Scout (vision).
# Fallback: HuggingFace Qwen2-VL if Groq fails or USE_HF_FALLBACK=true
# ─────────────────────────────────────────────────────────────

import os
import re
import json
import base64
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────
DETECTION_SYSTEM_PROMPT = """You are a precise food detection assistant.
Your job is to identify all food items visible in an image and return structured JSON.
Return ONLY valid JSON. No prose, no markdown, no explanation."""

DETECTION_USER_PROMPT = """Analyze this image and list every food item visible.

Return a JSON array where each object has exactly these fields:
- food_name: string (specific food name, e.g. "samosa" not "snack")
- portion_size: float (estimated quantity)
- portion_unit: string (one of: "piece", "cup", "bowl", "grams", "slice", "tablespoon")
- confidence: float between 0 and 1 (how certain you are about this item)
- visual_description: string (one sentence describing what you see)

Example output:
[
  {
    "food_name": "dal tadka",
    "portion_size": 1,
    "portion_unit": "bowl",
    "confidence": 0.92,
    "visual_description": "A bowl of yellow lentil dal with tempering visible on top."
  },
  {
    "food_name": "roti",
    "portion_size": 2,
    "portion_unit": "piece",
    "confidence": 0.95,
    "visual_description": "Two whole wheat rotis on the side of the plate."
  }
]

If no food is visible, return an empty array: []
Return ONLY the JSON array. Nothing else."""


def detect_foods(image_bytes: bytes, mime_type: str = "image/jpeg") -> list[dict]:
    """
    Detect food items from image bytes.

    Returns list of dicts with keys:
        food_name, portion_size, portion_unit, confidence, visual_description

    Raises:
        RuntimeError if both Groq and HuggingFace fallback fail
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{image_b64}"

    use_hf = os.getenv("USE_HF_FALLBACK", "false").lower() == "true"

    if not use_hf:
        try:
            logger.info("[vision] Calling Groq Llama 4 Scout for food detection")
            result = _groq_detect(data_url)
            logger.info(f"[vision] Groq detected {len(result)} item(s)")
            return result
        except Exception as e:
            logger.warning(f"[vision] Groq detection failed: {e} — switching to HuggingFace fallback")

    # HuggingFace fallback
    logger.info("[vision] Using HuggingFace Qwen2-VL fallback")
    return _hf_detect(image_b64)


def _groq_detect(data_url: str) -> list[dict]:
    """Call Groq Llama 4 Scout with image."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": DETECTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": DETECTION_USER_PROMPT},
                ],
            },
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    logger.debug(f"[vision] Groq raw response: {raw[:300]}...")
    return _parse_detection_response(raw)


def _hf_detect(image_b64: str) -> list[dict]:
    """HuggingFace Qwen2-VL fallback."""
    import requests

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN not set. Add it to your .env file to use HuggingFace fallback."
        )

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": {
            "image": image_b64,
            "text": DETECTION_USER_PROMPT,
        }
    }

    resp = requests.post(
        "https://api-inference.huggingface.co/models/Qwen/Qwen2-VL-7B-Instruct",
        headers=headers,
        json=payload,
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text[:200]}")

    raw = resp.json()
    # HF returns list with generated_text key
    text = raw[0].get("generated_text", "") if isinstance(raw, list) else str(raw)
    logger.debug(f"[vision] HuggingFace raw response: {text[:300]}...")
    return _parse_detection_response(text)


def _parse_detection_response(raw: str) -> list[dict]:
    """
    Parse model output into structured list.
    Three-layer fallback:
      1. Direct json.loads
      2. Regex extraction of JSON array
      3. Ask Groq text model to fix malformed JSON
    """
    # Layer 1: direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return _validate_items(data)
    except json.JSONDecodeError:
        logger.debug("[vision] Direct JSON parse failed, trying regex")

    # Layer 2: regex extraction
    try:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if isinstance(data, list):
                return _validate_items(data)
    except json.JSONDecodeError:
        logger.debug("[vision] Regex extraction failed, trying Groq reformat")

    # Layer 3: ask Groq to fix it
    try:
        logger.warning("[vision] Attempting Groq JSON reformat as last resort")
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        fix_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You fix malformed JSON. Return only valid JSON, nothing else."},
                {"role": "user", "content": f"Fix this into a valid JSON array of food objects:\n{raw}"},
            ],
            max_tokens=1024,
            temperature=0,
        )
        fixed = fix_response.choices[0].message.content.strip()
        data = json.loads(fixed)
        if isinstance(data, list):
            return _validate_items(data)
    except Exception as e:
        logger.error(f"[vision] All JSON parsing layers failed: {e}")

    logger.error("[vision] Could not parse model response — returning empty list")
    return []


def _validate_items(items: list) -> list[dict]:
    """Ensure each item has required fields with sensible defaults."""
    valid = []
    required = ["food_name", "portion_size", "portion_unit", "confidence", "visual_description"]
    valid_units = {"piece", "cup", "bowl", "grams", "slice", "tablespoon"}

    for item in items:
        if not isinstance(item, dict):
            logger.warning(f"[vision] Skipping non-dict item: {item}")
            continue

        # Apply defaults for missing fields
        item.setdefault("food_name", "unknown food")
        item.setdefault("portion_size", 1.0)
        item.setdefault("portion_unit", "piece")
        item.setdefault("confidence", 0.5)
        item.setdefault("visual_description", "No description available")

        # Clamp confidence to 0-1
        item["confidence"] = max(0.0, min(1.0, float(item["confidence"])))

        # Normalise unit
        if item["portion_unit"] not in valid_units:
            logger.warning(f"[vision] Unknown unit '{item['portion_unit']}' for '{item['food_name']}' — defaulting to 'piece'")
            item["portion_unit"] = "piece"

        # Convert portion_size to float
        try:
            item["portion_size"] = float(item["portion_size"])
        except (ValueError, TypeError):
            item["portion_size"] = 1.0

        valid.append(item)
        logger.debug(f"[vision] Validated item: {item['food_name']} | {item['portion_size']} {item['portion_unit']} | conf={item['confidence']}")

    return valid
