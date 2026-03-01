# utils/label_ocr.py
# ─────────────────────────────────────────────────────────────
# Food label OCR using Groq Llama 4 Scout (vision).
# Extracts structured nutrition facts from a label photo.
# ─────────────────────────────────────────────────────────────

import os
import re
import json
import base64
import logging
from groq import Groq

logger = logging.getLogger(__name__)

LABEL_SYSTEM_PROMPT = """You are a nutrition label extraction assistant.
Extract nutrition data from food labels and return ONLY valid JSON.
No prose, no markdown, no explanation."""

LABEL_USER_PROMPT = """Extract all nutrition information from this food label image.

Return ONLY this JSON structure (use null for any field not visible on the label):
{
  "product_name": "string or null",
  "serving_size_g": number or null,
  "per_serving": {
    "calories_kcal": number or null,
    "total_fat_g": number or null,
    "saturated_fat_g": number or null,
    "trans_fat_g": number or null,
    "carbohydrates_g": number or null,
    "fiber_g": number or null,
    "sugar_g": number or null,
    "protein_g": number or null,
    "sodium_mg": number or null,
    "calcium_mg": number or null,
    "iron_mg": number or null,
    "potassium_mg": number or null,
    "vit_a_mcg": number or null,
    "vit_c_mg": number or null,
    "vit_d_mcg": number or null
  },
  "label_quality": "clear / angled / partial / low_contrast",
  "notes": "any important observations about the label"
}

If the label is in a non-English language, still extract the numbers and note the language.
Return ONLY the JSON object."""


def extract_label(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    """
    Extract structured nutrition facts from a food label image.

    Returns:
        dict with keys: product_name, serving_size_g, per_serving, label_quality, notes
        Returns empty dict with error key if extraction fails.
    """
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{image_b64}"

    logger.info("[label_ocr] Sending label image to Groq Llama 4 Scout")

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": LABEL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": LABEL_USER_PROMPT},
                    ],
                },
            ],
            max_tokens=1024,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()
        logger.debug(f"[label_ocr] Raw response: {raw[:400]}...")
        result = _parse_label_response(raw)
        logger.info(f"[label_ocr] Extracted label: quality={result.get('label_quality')}, product={result.get('product_name')}")
        return result

    except Exception as e:
        logger.error(f"[label_ocr] Groq label extraction failed: {e}")
        return {"error": str(e), "per_serving": {}}


def serving_calculator(per_serving: dict, servings_consumed: float) -> dict:
    """
    Scale per-serving nutrition values by number of servings consumed.

    Args:
        per_serving: dict of nutrients per serving
        servings_consumed: how many servings were eaten (e.g. 1.5)

    Returns:
        Scaled nutrient dict in same format as meal analysis output
    """
    if not per_serving:
        logger.warning("[label_ocr] Empty per_serving dict passed to serving_calculator")
        return {}

    scaled = {}
    for key, value in per_serving.items():
        if value is None:
            scaled[key] = None
        else:
            try:
                scaled[key] = round(float(value) * servings_consumed, 2)
            except (TypeError, ValueError):
                scaled[key] = None

    # Map label keys to meal output schema for UI consistency
    return {
        "calories":        scaled.get("calories_kcal"),
        "fat_g":           scaled.get("total_fat_g"),
        "saturated_fat_g": scaled.get("saturated_fat_g"),
        "carbs_g":         scaled.get("carbohydrates_g"),
        "fiber_g":         scaled.get("fiber_g"),
        "sugar_g":         scaled.get("sugar_g"),
        "protein_g":       scaled.get("protein_g"),
        "sodium_mg":       scaled.get("sodium_mg"),
        "calcium_mg":      scaled.get("calcium_mg"),
        "iron_mg":         scaled.get("iron_mg"),
        "potassium_mg":    scaled.get("potassium_mg"),
        "vit_a_mcg":       scaled.get("vit_a_mcg"),
        "vit_c_mg":        scaled.get("vit_c_mg"),
        "vit_d_mcg":       scaled.get("vit_d_mcg"),
        "_servings_consumed": servings_consumed,
        "_is_label_mode": True,
    }


def _parse_label_response(raw: str) -> dict:
    """Parse Groq label response with fallback layers."""
    # Layer 1: direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "per_serving" in data:
            return data
    except json.JSONDecodeError:
        logger.debug("[label_ocr] Direct JSON parse failed, trying regex")

    # Layer 2: extract JSON object via regex
    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if isinstance(data, dict):
                data.setdefault("per_serving", {})
                return data
    except json.JSONDecodeError:
        logger.debug("[label_ocr] Regex extraction failed")

    # Layer 3: return empty with note
    logger.error("[label_ocr] Could not parse label response")
    return {
        "error": "Could not parse label data. Try a clearer, flatter photo.",
        "per_serving": {},
        "label_quality": "unreadable",
        "notes": raw[:200],  # include raw for debugging
    }
