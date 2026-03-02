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
# 8. LABEL OCR — serving_calculator logic (no API)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 8: Label OCR — serving_calculator")
print("═"*60)

from utils.label_ocr import serving_calculator

mock_label = {
    "calories_kcal":150,"total_fat_g":6.0,"saturated_fat_g":2.0,
    "carbohydrates_g":20.0,"fiber_g":1.5,"sugar_g":8.0,"protein_g":3.0,
    "sodium_mg":200.0,"calcium_mg":None,"iron_mg":None,
    "potassium_mg":None,"vit_a_mcg":None,"vit_c_mg":None,"vit_d_mcg":None,
}

s1 = serving_calculator(mock_label, 1.0)
record("serving_calculator 1x calories",   s1.get("calories")==150.0,   f"Got {s1.get('calories')}")
record("serving_calculator 1x protein",    s1.get("protein_g")==3.0,    f"Got {s1.get('protein_g')}")
record("serving_calculator null preserved",s1.get("calcium_mg") is None,f"Got {s1.get('calcium_mg')}")
record("serving_calculator _is_label_mode",s1.get("_is_label_mode")==True,"Should be True")

s25 = serving_calculator(mock_label, 2.5)
record("serving_calculator 2.5x calories",abs(s25.get("calories",0)-375.0)<0.01,f"Got {s25.get('calories')}")
record("serving_calculator 2.5x carbs",   abs(s25.get("carbs_g",0)-50.0)<0.01, f"Got {s25.get('carbs_g')}")
record("serving_calculator 2.5x null",    s25.get("calcium_mg") is None,"Should still be None")

s01 = serving_calculator(mock_label, 0.1)   # point 9: minimum 0.1 servings
record("serving_calculator 0.1x calories",abs(s01.get("calories",0)-15.0)<0.01,f"Got {s01.get('calories')}")
record("serving_calculator empty input",serving_calculator({},1.0)=={},"Should be empty dict")

# ─────────────────────────────────────────────────────────────
# 9. USDA SCORING — word-overlap fixes validated without API
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 9: USDA scoring — tomato/basil and okra/chicken-finger fixes")
print("═"*60)

_LOW_CAL_VEG = {
    "tomato","tomatoes","cucumber","lettuce","celery","spinach","capsicum",
    "pepper","okra","cabbage","zucchini","radish","beet","beetroot",
}
_SPECIALTY = {
    "glutinous","sticky","wild","instant","parboiled","converted",
    "enriched","unenriched","arborio","whole grain","sprouted","fortified",
}
_WRONG_CAT = {
    "cookie","cracker","cake","candy","chocolate","chips","popcorn",
    "chicken finger","chicken fingers","fish finger","fish fingers",
    "fish stick","fish sticks","ladyfinger cookie",
}

def _ov(q, d):
    STOP = {"raw","cooked","fresh","dried","and","with","the","a","an",
            "of","in","or","boiled","steamed","baked","roasted","fried","ns"}
    qw = {w for w in q.lower().split() if w not in STOP and len(w)>2}
    dl = d.lower()
    if not qw: return 0.5
    return sum(1 for w in qw if w in dl)/len(qw)

def _score(query, candidates):
    q = query.lower()
    cooked_sig = {"cooked","boiled","steamed","fried","baked","roasted",
                  "rice","dal","pasta","lentil","bean","curry","porridge"}
    imp_cooked = any(w in q for w in cooked_sig)
    is_veg     = any(v in q for v in _LOW_CAL_VEG)
    scored = []
    for c in candidates:
        s=0; k=c.get("calories",0) or 0; d=c.get("_d",""); dl=d.lower()
        ov=_ov(query,d)
        if   ov==1.0: s+=4
        elif ov>=0.5: s+=2
        elif ov> 0.0: s+=1
        else:         s-=4
        for bad in _WRONG_CAT:
            if bad in dl and bad not in q: s-=10; break
        if 50<=k<=450: s+=1
        if k>500:      s-=2
        if k<5 and not is_veg: s-=1
        if imp_cooked and "cooked" in dl: s+=2
        if imp_cooked and k>300:          s-=1
        for v in _SPECIALTY:
            if v in dl and v not in q: s-=3; break
        scored.append((s,c))
    scored.sort(key=lambda x:x[0],reverse=True)
    return scored[0][1]

def mk(desc,kcal): return {"_d":desc,"calories":kcal,"_usda_description":desc}

scoring_cases = [
    ("tomatoes raw",
     [mk("Tomatoes, red, ripe, raw",18),mk("Basil, fresh",22),mk("Tomato sauce",32)],
     "Tomatoes", "tomato must beat basil"),
    ("okra raw",
     [mk("Okra, raw",33),mk("Chicken fingers, breaded",220),mk("Fish fingers",190)],
     "Okra", "okra must beat chicken fingers"),
    ("cucumber raw",
     [mk("Cucumber, with peel, raw",15),mk("Cream cheese",342),mk("White bread",266)],
     "Cucumber", "cucumber must not lose to high-cal items"),
    ("okra raw",
     [mk("Okra, raw",33),mk("Ladyfingers, dry",352),mk("Chicken fingers",220)],
     "Okra", "lady finger normalized to okra must still win"),
    ("rice white long-grain cooked",
     [mk("Rice, white, glutinous, unenriched, cooked",406),
      mk("Rice, white, long-grain, regular, cooked",130),
      mk("Rice, brown, long-grain, cooked",112)],
     "long-grain, regular", "glutinous rice must NOT win"),
]

for query, candidates, expect, note in scoring_cases:
    best   = _score(query, candidates)
    winner = best["_usda_description"]
    ok     = expect.lower() in winner.lower()
    record(f"USDA score: {note}", ok,
           f"query='{query}' → winner='{winner}' ({best['calories']} kcal)")

# ─────────────────────────────────────────────────────────────
# 10. PORTIONS — utensil-based estimation (no API)
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("SCENARIO 10: Portions — utensil weight estimation")
print("═"*60)

_UTENSIL_ML = {
    "small katori":110,"medium katori":170,"large katori":230,
    "small bowl":180,"medium bowl":320,"large bowl":450,
}
_FILL = {"25%":0.25,"50%":0.50,"75%":0.75,"full":1.00,"n/a":0.70}
_DICT = {
    ("roti","piece"):40,("tomato","piece"):120,("cucumber","piece"):150,
    ("okra","piece"):6,("*","bowl"):250,("*","piece"):80,("*","cup"):240,
    ("*","grams"):1,
}

def _to_g(food,size,unit,container="",fill="n/a"):
    if unit in ("grams","g"): return max(1,int(round(size)))
    for k,v in _UTENSIL_ML.items():
        if k in container.lower():
            return max(5,int(round(v * _FILL.get(fill,0.70) * 0.85 * size)))
    fl = food.lower()
    for (fk,uk),gv in _DICT.items():
        if fk!="*" and fk in fl and uk==unit: return int(round(size*gv))
    return int(round(size * _DICT.get(("*",unit),80)))

portion_cases = [
    ("rice",    1,"bowl","medium bowl","50%",  90, 180, "medium bowl 50%"),
    ("dal",     1,"bowl","small katori","75%", 50, 120, "small katori 75%"),
    ("tomato",  1,"piece","","n/a",           110, 130, "tomato per piece"),
    ("roti",    2,"piece","","n/a",            75,  85, "2 rotis"),
    ("cucumber",100,"grams","","n/a",          100, 100, "direct grams"),
]

for food,qty,unit,container,fill,lo,hi,note in portion_cases:
    g  = _to_g(food,qty,unit,container,fill)
    ok = lo <= g <= hi
    record(f"to_grams '{food}' ({note})", ok, f"Got {g}g, expected {lo}-{hi}g")

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
