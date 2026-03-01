# utils/nutrition.py
# ─────────────────────────────────────────────────────────────
# Nutrition data resolution pipeline.
# Priority: IFCT (local) → USDA API → Open Food Facts API
# ─────────────────────────────────────────────────────────────

import os
import json
import logging
import requests
from pathlib import Path
from fuzzywuzzy import process as fw_process

logger = logging.getLogger(__name__)

# ── RDI Reference values (adult, per day) ─────────────────────
RDI = {
    "calories":    2000,
    "carbs_g":      275,
    "fiber_g":       28,
    "sugar_g":       50,
    "protein_g":     50,
    "fat_g":         78,
    "saturated_fat_g": 20,
    "vit_a_mcg":    900,
    "vit_c_mg":      90,
    "vit_d_mcg":     20,
    "vit_e_mg":      15,
    "vit_k_mcg":    120,
    "vit_b1_mg":    1.2,
    "vit_b2_mg":    1.3,
    "vit_b3_mg":     16,
    "vit_b6_mg":    1.7,
    "vit_b12_mcg":  2.4,
    "iron_mg":       18,
    "calcium_mg":  1000,
    "zinc_mg":       11,
    "magnesium_mg": 420,
    "potassium_mg":4700,
    "sodium_mg":   2300,
    "phosphorus_mg":700,
}

# ── Load IFCT dataset once at import ─────────────────────────
_IFCT_DATA = None

def _load_ifct() -> dict:
    global _IFCT_DATA
    if _IFCT_DATA is None:
        ifct_path = Path(__file__).parent.parent / "data" / "ifct.json"
        if ifct_path.exists():
            with open(ifct_path, "r", encoding="utf-8") as f:
                _IFCT_DATA = json.load(f)
            logger.info(f"[nutrition] IFCT loaded: {len(_IFCT_DATA)} entries")
        else:
            logger.warning(
                "[nutrition] IFCT data file not found at data/ifct.json — "
                "download it from GitHub (search: IFCT2017 json). "
                "Falling back to USDA for all lookups."
            )
            _IFCT_DATA = {}
    return _IFCT_DATA


# ── Main lookup function ──────────────────────────────────────

def lookup_nutrition(normalized_name: str) -> dict | None:
    """
    Look up nutrients per 100g for a food name.
    Returns a nutrient dict or None if all sources fail.
    """
    # 1. IFCT (local, instant)
    result = lookup_ifct(normalized_name)
    if result:
        result["_source"] = "IFCT"
        return result

    # 2. USDA API
    result = lookup_usda(normalized_name)
    if result:
        result["_source"] = "USDA"
        return result

    # 3. Open Food Facts
    result = lookup_openfoodfacts(normalized_name)
    if result:
        result["_source"] = "OpenFoodFacts"
        return result

    logger.warning(f"[nutrition] No data found for '{normalized_name}' in any source")
    return None


def lookup_ifct(name: str) -> dict | None:
    """Fuzzy-match against local IFCT JSON dataset."""
    data = _load_ifct()
    if not data:
        return None

    food_names = list(data.keys())
    match = fw_process.extractOne(name, food_names, score_cutoff=80)

    if match:
        matched_name, score, *_ = match
        logger.debug(f"[nutrition] IFCT match: '{name}' → '{matched_name}' (score={score})")
        return data[matched_name]
    else:
        logger.debug(f"[nutrition] IFCT no match for '{name}' (best score below 80)")
        return None


def lookup_usda(name: str) -> dict | None:
    """
    Query USDA FoodData Central API.

    Fetches top 5 results and picks the best match using:
    1. Prefer results whose description contains 'cooked' if query implies cooked food
    2. Prefer calorie range 50–600 kcal/100g (rejects raw grains ~360+ and water ~0)
    3. Falls back to first result if no candidate passes filters
    """
    api_key = os.getenv("USDA_API_KEY", "DEMO_KEY")
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"

    try:
        resp = requests.get(
            url,
            params={
                "query": name,
                "api_key": api_key,
                "dataType": "SR Legacy,Foundation",
                "pageSize": 5,   # fetch 5, pick best
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        foods = data.get("foods", [])
        if not foods:
            logger.debug(f"[nutrition] USDA: no results for '{name}'")
            return None

        # Build candidates with parsed nutrients
        candidates = []
        for food in foods:
            nutrient_map = {n["nutrientName"]: n["value"] for n in food.get("foodNutrients", [])}
            parsed = _parse_usda_nutrients(nutrient_map)
            parsed["_usda_description"] = food.get("description", "")
            candidates.append(parsed)
            logger.debug(
                f"[nutrition] USDA candidate: '{food.get('description')}' "
                f"→ {parsed.get('calories', '?')} kcal/100g"
            )

        # Pick best candidate
        best = _pick_best_usda_result(name, candidates)
        logger.debug(f"[nutrition] USDA selected: '{best.get('_usda_description')}' "
                     f"({best.get('calories')} kcal/100g)")
        return best

    except requests.Timeout:
        logger.error("[nutrition] USDA API timeout (10s)")
        return None
    except requests.HTTPError as e:
        logger.error(f"[nutrition] USDA HTTP error: {e}")
        return None
    except Exception as e:
        logger.error(f"[nutrition] USDA unexpected error: {e}")
        return None


def _pick_best_usda_result(query: str, candidates: list[dict]) -> dict:
    """
    Score USDA candidates and return the most appropriate one.

    Scoring rules:
    - +2 if description contains 'cooked' and query implies cooked food
    - +1 if calories are in realistic cooked-food range (50–250 kcal/100g)
    - -2 if calories > 300 kcal/100g and query implies cooked food
    - -2 if calories > 500 kcal/100g (raw grain, nut butter, oil)
    - -1 if calories < 20 kcal/100g (broth, water)
    - -3 if description contains a specialty/variant keyword not in query
      (e.g. 'glutinous', 'wild', 'brown' when query just says 'rice')
    """
    query_lower = query.lower()

    # Words that imply the food should be cooked / prepared
    cooked_signals = {"cooked", "boiled", "steamed", "fried", "baked", "roasted",
                      "rice", "dal", "pasta", "lentil", "bean", "curry", "porridge"}
    query_implies_cooked = any(w in query_lower for w in cooked_signals)

    # Specialty/variant keywords — penalise if they appear in the result
    # description but NOT in the user's query (we don't want glutinous rice
    # when the user just searched for 'rice cooked white')
    specialty_variants = {
        "glutinous", "sticky", "wild", "instant", "parboiled",
        "converted", "enriched", "unenriched", "arborio",
        "whole grain", "sprouted", "fortified",
    }

    scored = []
    for c in candidates:
        score = 0
        kcal = c.get("calories", 0) or 0
        desc = c.get("_usda_description", "").lower()

        # Cooked match bonus
        if query_implies_cooked and "cooked" in desc:
            score += 2

        # Calorie range scoring
        if 50 <= kcal <= 250:
            score += 1
        if query_implies_cooked and kcal > 300:
            score -= 2   # cooked food should not be this dense
        if kcal > 500:
            score -= 2   # raw grain / oil
        if kcal < 20:
            score -= 1   # broth / water

        # Penalise specialty variants not mentioned in query
        for variant in specialty_variants:
            if variant in desc and variant not in query_lower:
                score -= 3
                logger.debug(f"[nutrition] Penalised '{desc}' for specialty variant '{variant}'")
                break  # only penalise once per candidate

        scored.append((score, c))
        logger.debug(
            f"[nutrition] Scored '{c.get('_usda_description')}': "
            f"{score} pts ({kcal} kcal/100g)"
        )

    # Sort by score descending; stable sort preserves USDA relevance rank for ties
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]

    # Final safety net: if best is still suspiciously high calorie for a
    # cooked-food query, log a loud warning so developer can investigate
    best_kcal = best.get("calories", 0) or 0
    if query_implies_cooked and best_kcal > 300:
        logger.warning(
            f"[nutrition] USDA best result for '{query}' is '{best.get('_usda_description')}' "
            f"at {best_kcal} kcal/100g — this may still be wrong. "
            f"Add '{query}' to normalize.py dict with a more specific USDA term."
        )

    return best


def lookup_openfoodfacts(name: str) -> dict | None:
    """Query Open Food Facts API."""
    try:
        resp = requests.get(
            "https://world.openfoodfacts.org/cgi/search.pl",
            params={
                "search_terms": name,
                "json": 1,
                "page_size": 1,
                "fields": "nutriments,product_name",
            },
            timeout=10,
        )
        resp.raise_for_status()
        products = resp.json().get("products", [])

        if not products:
            logger.debug(f"[nutrition] OpenFoodFacts: no results for '{name}'")
            return None

        n = products[0].get("nutriments", {})
        logger.debug(f"[nutrition] OpenFoodFacts matched '{name}' → '{products[0].get('product_name')}'")
        return {
            "calories":        n.get("energy-kcal_100g", 0),
            "carbs_g":         n.get("carbohydrates_100g", 0),
            "fiber_g":         n.get("fiber_100g", 0),
            "sugar_g":         n.get("sugars_100g", 0),
            "protein_g":       n.get("proteins_100g", 0),
            "fat_g":           n.get("fat_100g", 0),
            "saturated_fat_g": n.get("saturated-fat_100g", 0),
            "sodium_mg":       n.get("sodium_100g", 0) * 1000,
            "iron_mg":         n.get("iron_100g", 0) * 1000,
            "calcium_mg":      n.get("calcium_100g", 0) * 1000,
            "vit_c_mg":        n.get("vitamin-c_100g", 0) * 1000,
        }
    except requests.Timeout:
        logger.error("[nutrition] OpenFoodFacts API timeout")
        return None
    except Exception as e:
        logger.error(f"[nutrition] OpenFoodFacts error: {e}")
        return None


def scale_nutrients(nutrients_per_100g: dict, actual_grams: int) -> dict:
    """Scale all nutrient values from per-100g to actual portion."""
    factor = actual_grams / 100.0
    scaled = {}
    for key, value in nutrients_per_100g.items():
        if key.startswith("_"):          # skip metadata keys like _source
            scaled[key] = value
        else:
            try:
                scaled[key] = round(float(value) * factor, 2)
            except (TypeError, ValueError):
                scaled[key] = 0.0
    scaled["_grams"] = actual_grams
    return scaled


def aggregate_meal(scaled_items: list[dict]) -> dict:
    """
    Sum all per-item scaled nutrients into a meal total.
    Also calculates macro %calories and micro %RDI.
    """
    totals = {key: 0.0 for key in RDI}
    totals["_grams"] = 0

    for item in scaled_items:
        nutrients = item.get("nutrients", {})
        for key in totals:
            totals[key] += nutrients.get(key, 0.0)

    # Macro % of total calories
    kcal = totals["calories"] or 1  # avoid divide-by-zero
    totals["_carb_pct"]    = round((totals["carbs_g"]   * 4 / kcal) * 100, 1)
    totals["_protein_pct"] = round((totals["protein_g"] * 4 / kcal) * 100, 1)
    totals["_fat_pct"]     = round((totals["fat_g"]     * 9 / kcal) * 100, 1)

    # Micro %RDI
    for key, rdi_val in RDI.items():
        if key in ("calories", "carbs_g", "fiber_g", "sugar_g", "protein_g", "fat_g", "saturated_fat_g"):
            continue  # macros handled above
        rdi_key = f"_rdi_{key}"
        totals[rdi_key] = round((totals.get(key, 0) / rdi_val) * 100, 1) if rdi_val else 0

    return totals


def _parse_usda_nutrients(n: dict) -> dict:
    """Map USDA nutrient names to our standard keys."""
    def get(*keys):
        for k in keys:
            if k in n:
                return float(n[k])
        return 0.0

    return {
        "calories":        get("Energy"),
        "carbs_g":         get("Carbohydrate, by difference"),
        "fiber_g":         get("Fiber, total dietary"),
        "sugar_g":         get("Sugars, total including NLEA"),
        "protein_g":       get("Protein"),
        "fat_g":           get("Total lipid (fat)"),
        "saturated_fat_g": get("Fatty acids, total saturated"),
        "sodium_mg":       get("Sodium, Na"),
        "iron_mg":         get("Iron, Fe"),
        "calcium_mg":      get("Calcium, Ca"),
        "zinc_mg":         get("Zinc, Zn"),
        "magnesium_mg":    get("Magnesium, Mg"),
        "potassium_mg":    get("Potassium, K"),
        "phosphorus_mg":   get("Phosphorus, P"),
        "vit_a_mcg":       get("Vitamin A, RAE"),
        "vit_c_mg":        get("Vitamin C, total ascorbic acid"),
        "vit_d_mcg":       get("Vitamin D (D2 + D3)"),
        "vit_e_mg":        get("Vitamin E (alpha-tocopherol)"),
        "vit_k_mcg":       get("Vitamin K (phylloquinone)"),
        "vit_b1_mg":       get("Thiamin"),
        "vit_b2_mg":       get("Riboflavin"),
        "vit_b3_mg":       get("Niacin"),
        "vit_b6_mg":       get("Vitamin B-6"),
        "vit_b12_mcg":     get("Vitamin B-12"),
    }
