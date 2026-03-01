# utils/portions.py
# ─────────────────────────────────────────────────────────────
# Converts portion descriptions (e.g. "1 cup") to grams.
# Primary: static lookup dict (instant, no API call)
# Fallback: Groq Llama 3.3 70B estimate
# ─────────────────────────────────────────────────────────────

import os
import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# ── Static portion-to-grams lookup ───────────────────────────
# Format: (food_keyword_or_"*", unit) -> grams
# "*" means this unit applies to any food not specifically listed
PORTION_TO_GRAMS = {
    # ── Indian staples ────────────────────────────────────────
    ("roti",        "piece"):       40,
    ("chapati",     "piece"):       40,
    ("paratha",     "piece"):       80,
    ("naan",        "piece"):       90,
    ("puri",        "piece"):       30,
    ("idli",        "piece"):       40,
    ("dosa",        "piece"):       80,
    ("vada",        "piece"):       50,
    ("samosa",      "piece"):       60,
    ("kachori",     "piece"):       55,
    ("dhokla",      "piece"):       35,

    # ── Rice dishes ───────────────────────────────────────────
    ("rice",        "cup"):        186,
    ("rice",        "bowl"):       200,
    ("biryani",     "cup"):        200,
    ("biryani",     "bowl"):       280,
    ("pulao",       "cup"):        190,
    ("fried rice",  "cup"):        200,

    # ── Dal and curries ───────────────────────────────────────
    ("dal",         "bowl"):       250,
    ("dal",         "cup"):        200,
    ("curry",       "bowl"):       250,
    ("curry",       "cup"):        200,
    ("sabzi",       "bowl"):       200,
    ("chole",       "bowl"):       250,
    ("rajma",       "bowl"):       250,
    ("sambar",      "bowl"):       200,

    # ── Breakfast ─────────────────────────────────────────────
    ("poha",        "cup"):        160,
    ("poha",        "bowl"):       180,
    ("upma",        "cup"):        160,
    ("upma",        "bowl"):       200,
    ("oats",        "cup"):        240,
    ("cornflakes",  "cup"):         28,

    # ── Snacks ────────────────────────────────────────────────
    ("pakora",      "piece"):       30,
    ("biscuit",     "piece"):       10,
    ("cookie",      "piece"):       15,
    ("bread",       "slice"):       30,

    # ── Fruits ───────────────────────────────────────────────
    ("apple",       "piece"):      182,
    ("banana",      "piece"):      118,
    ("orange",      "piece"):      131,
    ("mango",       "piece"):      200,
    ("grapes",      "cup"):        150,

    # ── Proteins ─────────────────────────────────────────────
    ("egg",         "piece"):       50,
    ("chicken",     "piece"):      120,
    ("fish",        "piece"):      120,
    ("paneer",      "cup"):        226,
    ("paneer",      "piece"):       50,

    # ── Generic units (apply to anything not listed above) ────
    ("*",           "cup"):        240,
    ("*",           "bowl"):       250,
    ("*",           "tablespoon"):  15,
    ("*",           "teaspoon"):     5,
    ("*",           "slice"):       30,
    ("*",           "piece"):       80,
    ("*",           "grams"):        1,   # portion_size IS grams
}


def to_grams(food_name: str, portion_size: float, portion_unit: str) -> int:
    """
    Convert a portion description to grams.

    Returns an integer gram weight.
    Logs a warning if fallback (Groq) is used.
    """
    food_lower = food_name.lower().strip()
    unit_lower = portion_unit.lower().strip()

    # 1. If unit is already grams — portion_size IS grams
    if unit_lower in ("grams", "g", "gram"):
        grams = int(round(portion_size))
        logger.debug(f"[portions] '{food_name}' unit=grams → {grams}g (direct)")
        return grams

    # 2. Check specific food + unit match
    for (food_key, unit_key), gram_val in PORTION_TO_GRAMS.items():
        if food_key == "*":
            continue
        if food_key in food_lower and unit_key == unit_lower:
            grams = int(round(portion_size * gram_val))
            logger.debug(f"[portions] '{food_name}' {portion_size} {portion_unit} → {grams}g (dict match: {food_key})")
            return grams

    # 3. Fall back to generic unit
    generic_key = ("*", unit_lower)
    if generic_key in PORTION_TO_GRAMS:
        grams = int(round(portion_size * PORTION_TO_GRAMS[generic_key]))
        logger.debug(f"[portions] '{food_name}' {portion_size} {portion_unit} → {grams}g (generic unit fallback)")
        return grams

    # 4. Ask Groq Llama 3.3 70B as last resort
    logger.warning(f"[portions] No dict match for '{food_name}' {portion_size} {portion_unit} — asking Groq")
    return _groq_estimate(food_name, portion_size, portion_unit)


def _groq_estimate(food_name: str, portion_size: float, portion_unit: str) -> int:
    """Ask Groq Llama 3.3 70B to estimate grams. Returns integer."""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = (
            f"Estimate the weight in grams for: {portion_size} {portion_unit} of {food_name}. "
            "Reply with a single integer only. No explanation, no units, just the number."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        grams = int("".join(filter(str.isdigit, raw)) or "100")
        logger.info(f"[portions] Groq estimated {grams}g for '{food_name}' {portion_size} {portion_unit}")
        return grams
    except Exception as e:
        logger.error(f"[portions] Groq gram estimate failed: {e} — defaulting to 100g")
        return 100  # safe fallback
