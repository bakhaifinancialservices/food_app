# tests/test_scenarios.py
# ─────────────────────────────────────────────────────────────
# Manual + automated test scenarios for every pipeline layer.
# Run: python tests/test_scenarios.py
#
# These tests use REAL API calls — they verify the live pipeline.
# Not unit tests — integration tests.
# ─────────────────────────────────────────────────────────────

import os
import sys
import json
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tests")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

results = []

def record(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((status, name, detail))
    print(f"{status}  {name}")
    if detail:
        print(f"       → {detail}")


# ─────────────────────────────────────────────────────────────
# 1. ENVIRONMENT CHECKS (no API calls)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 1: Environment & API Keys")
print("═"*60)

groq_key = os.getenv("GROQ_API_KEY", "")
usda_key = os.getenv("USDA_API_KEY", "")
hf_token = os.getenv("HF_TOKEN", "")

record("GROQ_API_KEY is set", bool(groq_key), groq_key[:12] + "..." if groq_key else "MISSING")
record("USDA_API_KEY is set", bool(usda_key), usda_key[:8] + "..." if usda_key else "Will use DEMO_KEY (limited)")
record("HF_TOKEN is set (optional)", bool(hf_token) or True, "Optional — only needed for HF fallback")


# ─────────────────────────────────────────────────────────────
# 2. PORTIONS (no API call)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 2: Portion-to-Grams Conversion (No API)")
print("═"*60)

from utils.portions import to_grams

test_cases = [
    ("roti",   1, "piece",       40,   "exact dict match"),
    ("rice",   1, "cup",         186,  "exact dict match"),
    ("dal",    1, "bowl",        250,  "exact dict match"),
    ("banana", 1, "piece",       118,  "exact dict match"),
    ("pasta",  2, "cup",         480,  "generic cup × 2"),
    ("roti",   2, "piece",       80,   "portion_size multiplication"),
    ("rice",  50, "grams",       50,   "grams passthrough"),
]

for food, qty, unit, expected, note in test_cases:
    result = to_grams(food, qty, unit)
    ok = (result == expected)
    record(f"to_grams('{food}', {qty}, '{unit}')", ok, f"Got {result}g, expected {expected}g ({note})")


# ─────────────────────────────────────────────────────────────
# 3. NORMALIZATION (Groq API — text only, fast)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 3: Food Name Normalization (Groq text API)")
print("═"*60)

from utils.normalize import normalize_name, is_vague

norm_cases = [
    ("roti",          "chapati whole wheat",   "dict exact"),
    ("dal",           "lentils cooked",        "dict exact"),
    ("paneer",        "cottage cheese",        "dict exact"),
    ("chole",         "chickpeas cooked",      "dict exact"),
    ("poha",          "flattened rice poha",   "dict exact"),
    ("gulab jamun",   "gulab jamun sweet",     "dict exact"),
]

for raw, expected, note in norm_cases:
    result = normalize_name(raw)
    ok = (result == expected)
    record(f"normalize_name('{raw}')", ok, f"Got '{result}', expected '{expected}' ({note})")

vague_cases = [
    ("curry",    True),
    ("dal",      False),
    ("food",     True),
    ("biryani",  False),
]
for name, expected in vague_cases:
    result = is_vague(name)
    record(f"is_vague('{name}')", result == expected, f"Got {result}, expected {expected}")


# ─────────────────────────────────────────────────────────────
# 4. NUTRITION LOOKUP (USDA API)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 4: Nutrition Lookup (USDA API)")
print("═"*60)

from utils.nutrition import lookup_nutrition, scale_nutrients, aggregate_meal

# Test known food — cooked white long-grain rice should be ~130 kcal per 100g
# Using the exact term normalize.py produces for "rice" to keep test in sync
result = lookup_nutrition("rice white long-grain cooked")
if result:
    kcal = result.get("calories", 0)
    usda_desc = result.get("_usda_description", "n/a")
    ok = 80 <= kcal <= 200
    record(
        "USDA lookup: 'rice white long-grain cooked'",
        ok,
        f"Calories: {kcal} kcal/100g | USDA matched: '{usda_desc}' | "
        f"{'GOOD - cooked range' if ok else 'BAD - still wrong variant, check logs'}",
    )
else:
    record("USDA lookup: 'rice white long-grain cooked'", False, "No data returned — check USDA_API_KEY")

# Test scaling: 1 cup cooked rice = 186g × ~130 kcal/100g ≈ 242 kcal
if result:
    scaled = scale_nutrients(result, 186)
    kcal_scaled = scaled.get("calories", 0)
    ok = 150 <= kcal_scaled <= 380
    record(
        "scale_nutrients: 186g rice",
        ok,
        f"Scaled: {kcal_scaled:.1f} kcal (expected 150–380 for cooked rice portion)",
    )

# Test Indian food fallback
result_paneer = lookup_nutrition("cottage cheese")
if result_paneer:
    record("Nutrition lookup: 'cottage cheese' (paneer)", True, f"Source: {result_paneer.get('_source')}, kcal: {result_paneer.get('calories')}")
else:
    record("Nutrition lookup: 'cottage cheese' (paneer)", False, "No data — check API keys")


# ─────────────────────────────────────────────────────────────
# 5. AGGREGATION (no API)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 5: Meal Aggregation (No API)")
print("═"*60)

mock_items = [
    {"food_name": "roti",   "nutrients": {"calories": 120, "carbs_g": 25, "protein_g": 3, "fat_g": 1}},
    {"food_name": "dal",    "nutrients": {"calories": 150, "carbs_g": 20, "protein_g": 10, "fat_g": 3}},
]

total = aggregate_meal(mock_items)
record("Meal calories summed correctly",  total.get("calories") == 270,  f"Got {total.get('calories')}, expected 270")
record("Macro %calories calculated",      "_carb_pct" in total,          f"carb%={total.get('_carb_pct')}, protein%={total.get('_protein_pct')}, fat%={total.get('_fat_pct')}")


# ─────────────────────────────────────────────────────────────
# 6. VALIDATION LOGIC (no API for layers 1 + 5)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 6: Validation Layers (partial — no API)")
print("═"*60)

from utils.validate import flag_low_confidence, flag_vague_terms, nutrition_sanity_check

items_conf = [
    {"food_name": "roti",  "confidence": 0.95},  # should NOT flag
    {"food_name": "rice",  "confidence": 0.60},  # SHOULD flag
    {"food_name": "curry", "confidence": 0.80},  # vague — should flag
]

flagged = flag_low_confidence(items_conf)
record("Low confidence flag (0.60)", flagged[1].get("is_flagged") == True, "rice at 0.60 should be flagged")
record("High confidence not flagged (0.95)", not flagged[0].get("is_flagged"), "roti at 0.95 should NOT be flagged")

flagged = flag_vague_terms(flagged)
record("Vague term 'curry' flagged", flagged[2].get("is_flagged") == True, "curry is in vague terms list")

sanity_items = [{"food_name": "magic rice", "nutrients": {"calories": 2000}}]  # impossibly high
sanity_total = {"calories": 2000}
warnings = nutrition_sanity_check(sanity_items, sanity_total)
record("Sanity check: 2000 kcal single item flagged", len(warnings) > 0, f"Warning: {warnings[0] if warnings else 'none'}")


# ─────────────────────────────────────────────────────────────
# 7. GROQ VISION MOCK (verifies SDK + model availability)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 7: Groq API Connectivity Check")
print("═"*60)

try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    # Simple text ping
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Reply with the word CONNECTED only."}],
        max_tokens=5,
    )
    reply = resp.choices[0].message.content.strip()
    record("Groq API connectivity (Llama 3.3 70B)", "CONNECTED" in reply.upper(), f"Response: '{reply}'")
except Exception as e:
    record("Groq API connectivity", False, str(e))

# Scout model availability check (text mode)
try:
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": "Reply OK"}],
        max_tokens=5,
    )
    record("Groq Llama 4 Scout available", True, "Model is accessible")
except Exception as e:
    record("Groq Llama 4 Scout available", False, f"Error: {e}")


# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("TEST SUMMARY")
print("═"*60)

passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total_tests = len(results)

print(f"\nTotal: {total_tests}  |  {PASS}: {passed}  |  {FAIL}: {failed}")

if failed > 0:
    print("\nFailed tests:")
    for status, name, detail in results:
        if status == FAIL:
            print(f"  {FAIL}  {name}")
            if detail:
                print(f"         {detail}")

print("\n" + "═"*60)
sys.exit(0 if failed == 0 else 1)
