# utils/validate.py — 4-layer hallucination detection & validation
#
# Layer 1: confidence threshold flagging
# Layer 2: Groq consistency check (parallel thread)
# Layer 3: nutrition sanity check (post-lookup)
# Layer 4: user disambiguation (UI handles this)
#
# Key rule: items with confidence >= 0.85 are NEVER forced into
# disambiguation even if flagged — they get a soft warning only.

import os, json, logging, threading
from groq import Groq
logger = logging.getLogger(__name__)

CONFIDENCE_FLAG      = 0.65   # below → auto-flagged
CONFIDENCE_DISAMBIG  = 0.85   # below → disambiguation offered; at/above → soft warning only
SINGLE_KCAL_MAX      = 1500
MICRO_RDI_MAX_PCT    = 500
MEAL_KCAL_MAX        = 5000

VAGUE_TERMS = {
    "curry","food","dish","meal","item","stuff","sauce","thing",
    "indian food","rice dish","bread","vegetable","fruit","meat",
    "snack","dessert","drink","beverage","mixed","unknown","various",
}


def run_validation(detected: list[dict]) -> tuple[list[dict], dict]:
    """Run all validation layers. Returns (enriched_items, consistency_flags)."""
    consistency_flags = {}
    thread_result     = {}

    def _bg():
        thread_result["flags"] = groq_consistency_check(detected)

    t = threading.Thread(target=_bg)
    t.start()

    enriched = flag_low_confidence(detected)
    enriched = flag_vague_terms(enriched)

    t.join(timeout=15)
    if "flags" in thread_result:
        consistency_flags = thread_result["flags"]
        enriched = apply_consistency_flags(enriched, consistency_flags)
    else:
        logger.warning("[validate] Consistency check timed out")

    return enriched, consistency_flags


def flag_low_confidence(items: list[dict]) -> list[dict]:
    for item in items:
        conf = item.get("confidence", 0)
        item.setdefault("is_flagged",            False)
        item.setdefault("needs_disambiguation",   False)
        item.setdefault("flag_reasons",           [])
        item.setdefault("disambiguation_options", [item.get("food_name", "")])

        if conf < CONFIDENCE_FLAG:
            item["is_flagged"]          = True
            item["needs_disambiguation"] = True
            item["flag_reasons"].append(
                f"Low confidence ({conf:.0%}) — please verify this item"
            )
            logger.debug(f"[validate] Low-conf flag: '{item['food_name']}' {conf:.0%}")
        elif conf < CONFIDENCE_DISAMBIG:
            item["needs_disambiguation"] = True
            item["flag_reasons"].append(
                f"Medium confidence ({conf:.0%}) — please confirm"
            )
    return items


def flag_vague_terms(items: list[dict]) -> list[dict]:
    for item in items:
        if item["food_name"].lower().strip() in VAGUE_TERMS:
            item["is_flagged"]          = True
            item["needs_disambiguation"] = True
            item["flag_reasons"].append(
                "Name is too generic — please specify the exact food"
            )
            logger.debug(f"[validate] Vague term: '{item['food_name']}'")
    return items


def groq_consistency_check(items: list[dict]) -> dict:
    """Ask Groq to flag implausible food combinations or portion sizes."""
    if not items:
        return {}
    summary = [
        {"name": i["food_name"],
         "portion": f"{i['portion_size']} {i['portion_unit']}",
         "confidence": i.get("confidence", 0)}
        for i in items
    ]
    try:
        client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt  = (
            "You are a food plausibility checker. "
            "Review this list of detected food items from one meal photo.\n\n"
            f"{json.dumps(summary, indent=2)}\n\n"
            "For each item, decide if it is plausible (correct food name, realistic portion).\n"
            "Flag items that look wrong — e.g. a wildly incorrect portion (360g of okra in a salad bowl), "
            "a food that doesn't belong (basil in a rice meal), or a completely wrong identification.\n"
            "Do NOT flag items just because they are unusual combinations — Indian meals often mix many foods.\n"
            "Return ONLY a JSON object: {food_name: {flagged: bool, reason: string}}"
        )
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512, temperature=0,
        )
        raw = r.choices[0].message.content.strip()
        # extract JSON object
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except Exception as e:
        logger.warning(f"[validate] Consistency check error: {e}")
        return {}


def apply_consistency_flags(items: list[dict], flags: dict) -> list[dict]:
    """
    Apply consistency check flags.
    HIGH confidence (>= 0.85): soft warning only — no forced disambiguation.
    Lower confidence: set is_flagged + needs_disambiguation.
    """
    for item in items:
        fd = flags.get(item["food_name"], {})
        if fd.get("flagged"):
            reason = fd.get("reason", "Flagged by consistency check")
            item["flag_reasons"].append(f"⚠ Consistency: {reason}")
            if item.get("confidence", 0) >= 0.85:
                logger.info(
                    f"[validate] High-conf item '{item['food_name']}' flagged — soft warning only"
                )
            else:
                item["is_flagged"]          = True
                item["needs_disambiguation"] = True
    return items


def nutrition_sanity_check(items: list[dict], meal_total: dict) -> list[str]:
    warnings = []
    for item in items:
        n    = item.get("nutrients", {})
        kcal = n.get("calories", 0) or 0
        name = item.get("food_name", "item")
        if kcal > SINGLE_KCAL_MAX:
            warnings.append(
                f"'{name}' shows {kcal:.0f} kcal — this seems very high. "
                f"Check the portion size."
            )
    meal_kcal = meal_total.get("calories", 0) or 0
    if meal_kcal > MEAL_KCAL_MAX:
        warnings.append(
            f"Total meal calories ({meal_kcal:.0f} kcal) seems very high. "
            f"Check portion sizes."
        )
    return warnings


def generate_disambiguation_options(food_name: str, visual_desc: str = "") -> list[str]:
    """Return 4 similar food name options via Groq."""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": (
                f"The food '{food_name}' was detected in an image. "
                f"Visual description: '{visual_desc}'. "
                "List 4 similar or alternative food names it could be "
                "(Indian foods prioritised). "
                "Return ONLY a JSON array of 4 strings."
            )}],
            max_tokens=100, temperature=0.3,
        )
        import re
        raw  = r.choices[0].message.content.strip()
        m    = re.search(r"\[.*\]", raw, re.DOTALL)
        opts = json.loads(m.group()) if m else []
        return [food_name] + [o for o in opts if o != food_name][:3]
    except Exception as e:
        logger.warning(f"[validate] Disambiguation generation failed: {e}")
        return [food_name]


def generate_all_disambiguation(items: list[dict]) -> list[dict]:
    """Pre-generate disambiguation options for all flagged items."""
    import concurrent.futures
    def _gen(item):
        if item.get("needs_disambiguation") or item.get("is_flagged"):
            item["disambiguation_options"] = generate_disambiguation_options(
                item["food_name"], item.get("visual_description", "")
            )
        return item
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(_gen, items))
