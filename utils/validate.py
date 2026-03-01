# utils/validate.py
# ─────────────────────────────────────────────────────────────
# 4-layer accuracy and hallucination detection system.
# All validation results are non-destructive — items are flagged,
# never silently removed. The user makes final decisions.
# ─────────────────────────────────────────────────────────────

import os
import json
import logging
import threading
from groq import Groq

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────
CONFIDENCE_FLAG_THRESHOLD   = 0.65   # below this → auto-flagged
CONFIDENCE_DISAMBIG_THRESHOLD = 0.75 # below this → disambiguation offered
SINGLE_ITEM_KCAL_MAX        = 1500   # kcal — above this is suspicious
MICRO_RDI_MAX_PCT           = 500    # % — above this is suspicious
MEAL_KCAL_MAX               = 5000   # kcal total — above this is suspicious

VAGUE_TERMS = {
    "curry", "food", "dish", "meal", "item", "stuff", "sauce",
    "thing", "indian food", "rice dish", "bread", "vegetable",
    "fruit", "meat", "snack", "dessert", "drink", "beverage",
    "mixed", "unknown", "various",
}


def run_validation(detected_items: list[dict]) -> tuple[list[dict], dict]:
    """
    Run all 4 validation layers on detected items.

    Returns:
        enriched_items: list with validation flags added
        consistency_flags: dict from Groq consistency check (keyed by food_name)
    """
    consistency_flags = {}

    # Layer 2: run Groq consistency check in background thread
    thread_result = {}
    def _run_consistency():
        thread_result["flags"] = groq_consistency_check(detected_items)

    thread = threading.Thread(target=_run_consistency)
    thread.start()

    # Layer 1: confidence filter (instant, no API)
    enriched = flag_low_confidence(detected_items)

    # Layer 5: vague term check (instant)
    enriched = flag_vague_terms(enriched)

    # Wait for Layer 2 to finish
    thread.join(timeout=15)  # max 15 seconds
    if "flags" in thread_result:
        consistency_flags = thread_result["flags"]
        enriched = apply_consistency_flags(enriched, consistency_flags)
    else:
        logger.warning("[validate] Groq consistency check timed out — skipping")

    # Generate disambiguation options for all flagged items
    enriched = generate_all_disambiguation(enriched)

    return enriched, consistency_flags


def flag_low_confidence(items: list[dict]) -> list[dict]:
    """Layer 1: Flag items with confidence below threshold."""
    for item in items:
        conf = item.get("confidence", 0)
        if conf < CONFIDENCE_FLAG_THRESHOLD:
            item["is_flagged"] = True
            item["flag_reasons"] = item.get("flag_reasons", [])
            item["flag_reasons"].append(f"Low confidence ({conf:.0%})")
            logger.debug(f"[validate] Low confidence flag: '{item['food_name']}' ({conf:.2f})")
        elif conf < CONFIDENCE_DISAMBIG_THRESHOLD:
            item["needs_disambiguation"] = True
            item["flag_reasons"] = item.get("flag_reasons", [])
            item["flag_reasons"].append(f"Moderate confidence ({conf:.0%}) — please verify")
    return items


def flag_vague_terms(items: list[dict]) -> list[dict]:
    """Flag items whose names are too generic to look up reliably."""
    for item in items:
        name = item.get("food_name", "").lower().strip()
        if name in VAGUE_TERMS:
            item["is_flagged"] = True
            item["needs_disambiguation"] = True
            item["flag_reasons"] = item.get("flag_reasons", [])
            item["flag_reasons"].append(f"Name '{item['food_name']}' is too generic — please specify")
            logger.warning(f"[validate] Vague term flagged: '{item['food_name']}'")
    return items


def groq_consistency_check(items: list[dict]) -> dict:
    """
    Layer 2: Ask Groq Llama 3.3 70B if the food combination is plausible.
    Returns dict keyed by food_name with {flagged, reason}.
    """
    if not items:
        return {}

    food_list = [item["food_name"] for item in items]
    logger.info(f"[validate] Running Groq consistency check for: {food_list}")

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = (
            f"These food items were detected in a single meal photo: {json.dumps(food_list)}\n\n"
            "For each item, determine if it is plausible to appear in a single Indian meal. "
            "Flag items that seem: (1) unlikely to appear together, (2) duplicate or redundant, "
            "(3) suspiciously generic.\n\n"
            "Return ONLY a JSON array like this:\n"
            '[{"food_name": "...", "flagged": false, "reason": ""}]\n'
            "Include ALL items in the array, even unflagged ones."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a food plausibility checker. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        result = {entry["food_name"]: entry for entry in data}
        flagged = [k for k, v in result.items() if v.get("flagged")]
        if flagged:
            logger.warning(f"[validate] Groq flagged as implausible: {flagged}")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"[validate] Groq consistency check JSON parse error: {e}")
        return {}
    except Exception as e:
        logger.error(f"[validate] Groq consistency check failed: {e}")
        return {}


def apply_consistency_flags(items: list[dict], flags: dict) -> list[dict]:
    """Apply Groq consistency check results to item list."""
    for item in items:
        flag_data = flags.get(item["food_name"], {})
        if flag_data.get("flagged"):
            item["is_flagged"] = True
            item["needs_disambiguation"] = True
            item["flag_reasons"] = item.get("flag_reasons", [])
            reason = flag_data.get("reason", "Flagged as implausible by consistency check")
            item["flag_reasons"].append(f"Consistency check: {reason}")
    return items


def nutrition_sanity_check(enriched_items: list[dict], meal_total: dict) -> list[str]:
    """
    Layer 3: Check for physically impossible nutrition values.
    Returns list of warning strings to display.
    """
    warnings = []

    for item in enriched_items:
        nutrients = item.get("nutrients", {})
        kcal = nutrients.get("calories", 0)
        if kcal > SINGLE_ITEM_KCAL_MAX:
            msg = f"'{item['food_name']}' shows {kcal:.0f} kcal — this seems high. Check the portion size."
            warnings.append(msg)
            logger.warning(f"[validate] Sanity check: {msg}")

        for key, val in nutrients.items():
            if key.startswith("_rdi_") and val > MICRO_RDI_MAX_PCT:
                nutrient_name = key.replace("_rdi_", "").replace("_", " ")
                msg = f"'{item['food_name']}' shows {val:.0f}% RDI for {nutrient_name} — likely a portion estimation error."
                warnings.append(msg)
                logger.warning(f"[validate] Sanity check: {msg}")

    total_kcal = meal_total.get("calories", 0)
    if total_kcal > MEAL_KCAL_MAX:
        msg = f"Total meal calories ({total_kcal:.0f} kcal) seems very high. Review portion sizes."
        warnings.append(msg)
        logger.warning(f"[validate] Sanity check: {msg}")

    return warnings


def generate_disambiguation_options(food_name: str, _cache: dict = {}) -> list[str]:
    """
    Generate 4 visually similar food alternatives via Groq.
    Results are cached in-memory to avoid repeated API calls.
    """
    if food_name in _cache:
        logger.debug(f"[validate] Disambiguation cache hit for '{food_name}'")
        return _cache[food_name]

    logger.info(f"[validate] Generating disambiguation options for '{food_name}'")
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You suggest alternative food names. Return only a JSON array of strings."},
                {
                    "role": "user",
                    "content": (
                        f"The detected food is '{food_name}' in an Indian meal context. "
                        "List exactly 4 foods that look visually similar or could easily be confused with it. "
                        f"Include '{food_name}' itself as one of the 4 options. "
                        "Return ONLY a JSON array of 4 strings, e.g.: [\"samosa\", \"kachori\", \"aloo tikki\", \"spring roll\"]"
                    ),
                },
            ],
            max_tokens=80,
            temperature=0.3,
        )
        raw = response.choices[0].message.content.strip()
        options = json.loads(raw)
        if isinstance(options, list) and len(options) > 0:
            # Ensure original is always included
            if food_name not in options:
                options[0] = food_name
            _cache[food_name] = options[:4]
            return _cache[food_name]
    except Exception as e:
        logger.error(f"[validate] Disambiguation option generation failed for '{food_name}': {e}")

    # Fallback: just return the original
    return [food_name]


def generate_all_disambiguation(items: list[dict]) -> list[dict]:
    """Pre-generate disambiguation options for all flagged items."""
    for item in items:
        if item.get("is_flagged") or item.get("needs_disambiguation"):
            item["disambiguation_options"] = generate_disambiguation_options(item["food_name"])
    return items
