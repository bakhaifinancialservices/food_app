# utils/nutrition.py — Nutrition lookup: IFCT → USDA → Open Food Facts
#
# KEY FIX: USDA candidate scoring now uses word-overlap as PRIMARY signal.
# This prevents tomato→basil and okra→chicken-finger mismatches.

import os, json, logging, requests
from pathlib import Path
from fuzzywuzzy import process as fw_process

logger = logging.getLogger(__name__)

# ── RDI (adult daily reference) ───────────────────────────────
RDI = {
    "calories":2000,"carbs_g":275,"fiber_g":28,"sugar_g":50,
    "protein_g":50,"fat_g":78,"saturated_fat_g":20,
    "vit_a_mcg":900,"vit_c_mg":90,"vit_d_mcg":20,"vit_e_mg":15,
    "vit_k_mcg":120,"vit_b1_mg":1.2,"vit_b2_mg":1.3,"vit_b3_mg":16,
    "vit_b6_mg":1.7,"vit_b12_mcg":2.4,
    "iron_mg":18,"calcium_mg":1000,"zinc_mg":11,"magnesium_mg":420,
    "potassium_mg":4700,"sodium_mg":2300,"phosphorus_mg":700,
}

# ── Low-calorie vegetables: NEVER penalise for being <20 kcal ─
# All raw values per 100g; tomato=18, cucumber=15, okra=31, etc.
LOW_CAL_VEGETABLES = {
    "tomato","tomatoes","cucumber","lettuce","celery","spinach","capsicum",
    "pepper","okra","cabbage","zucchini","radish","beet","beetroot",
    "asparagus","broccoli","cauliflower","mushroom","eggplant","aubergine",
    "kale","chard","arugula","rocket","endive","watercress","artichoke",
    "onion","leek","fennel","turnip","parsnip",
}

# ── Specialty variants: penalise if in USDA description but not in query ──
SPECIALTY_VARIANTS = {
    "glutinous","sticky","wild","instant","parboiled","converted",
    "enriched","unenriched","arborio","whole grain","sprouted","fortified",
}

# ── Wrong food category: hard-reject if query is vegetable/grain ──
WRONG_CATEGORY = {
    "cookie","cracker","cake","candy","chocolate","chips","popcorn",
    "pretzel","wafer","dessert","pastry","ladyfinger cookie",
    "chicken finger","chicken fingers","fish finger","fish fingers",
    "fish stick","fish sticks","mozzarella stick","onion ring",
}

_IFCT_CACHE = None


def _load_ifct() -> dict:
    global _IFCT_CACHE
    if _IFCT_CACHE is None:
        p = Path(__file__).parent.parent / "data" / "ifct.json"
        if p.exists():
            _IFCT_CACHE = json.loads(p.read_text("utf-8"))
            logger.info(f"[nutrition] IFCT loaded: {len(_IFCT_CACHE)} entries")
        else:
            logger.warning("[nutrition] data/ifct.json not found — IFCT disabled")
            _IFCT_CACHE = {}
    return _IFCT_CACHE


# ── Public API ────────────────────────────────────────────────

def lookup_nutrition(name: str) -> dict | None:
    """Try IFCT → USDA → Open Food Facts. Return first hit or None."""
    r = lookup_ifct(name)
    if r: r["_source"] = "IFCT"; return r
    r = lookup_usda(name)
    if r: r["_source"] = "USDA"; return r
    r = lookup_openfoodfacts(name)
    if r: r["_source"] = "OpenFoodFacts"; return r
    logger.warning(f"[nutrition] No data found for '{name}'")
    return None


def lookup_ifct(name: str) -> dict | None:
    data = _load_ifct()
    if not data: return None
    match = fw_process.extractOne(name, list(data.keys()), score_cutoff=80)
    if match:
        k, score, *_ = match
        logger.debug(f"[nutrition] IFCT: '{name}' → '{k}' (score={score})")
        return data[k]
    return None


def lookup_usda(name: str) -> dict | None:
    """Fetch top 10 USDA results, score with word-overlap + calorie plausibility."""
    try:
        resp = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={
                "query":    name,
                "api_key":  os.getenv("USDA_API_KEY", "DEMO_KEY"),
                "dataType": "SR Legacy,Foundation",
                "pageSize": 10,      # more candidates → better scoring
            },
            timeout=10,
        )
        resp.raise_for_status()
        foods = resp.json().get("foods", [])
        if not foods:
            logger.debug(f"[nutrition] USDA: no results for '{name}'")
            return None

        candidates = []
        for food in foods:
            nm = {n["nutrientName"]: n["value"] for n in food.get("foodNutrients", [])}
            parsed = _parse_usda_nutrients(nm)
            parsed["_usda_description"] = food.get("description", "")
            candidates.append(parsed)
            logger.debug(
                f"[nutrition] USDA candidate: '{food.get('description')}' "
                f"→ {parsed.get('calories','?')} kcal/100g"
            )

        best = _pick_best(name, candidates)
        logger.info(
            f"[nutrition] USDA selected: '{best.get('_usda_description')}' "
            f"({best.get('calories')} kcal/100g) for query='{name}'"
        )
        return best

    except requests.Timeout:
        logger.error("[nutrition] USDA timeout"); return None
    except requests.HTTPError as e:
        logger.error(f"[nutrition] USDA HTTP error: {e}"); return None
    except Exception as e:
        logger.error(f"[nutrition] USDA error: {e}"); return None


def _word_overlap(query: str, description: str) -> float:
    """
    PRIMARY matching signal — fraction of meaningful query words found in description.

    Stop-words (raw/cooked/fresh/and/with etc.) are excluded so they don't
    dilute the score. Result is 0.0 – 1.0.

    Examples:
      query='tomatoes raw', desc='Tomatoes, red, ripe, raw'
        meaningful words = {'tomatoes'} → 'tomatoes' in desc → 1.0  ✅

      query='tomatoes raw', desc='Basil, fresh'
        meaningful words = {'tomatoes'} → not in desc → 0.0  → -4 pts  ✅ rejected

      query='okra raw', desc='Chicken fingers, breaded'
        meaningful words = {'okra'} → not in desc → 0.0  → -4 pts  ✅ rejected

      query='rice white long-grain cooked', desc='Rice, white, long-grain, cooked'
        meaningful words = {'rice','white','long-grain'} → all found → 1.0  ✅
    """
    STOP = {
        "raw","cooked","fresh","dried","and","with","the","a","an",
        "of","in","or","boiled","steamed","baked","roasted","fried",
        "ns","nfs","upc",
    }
    query_words = {w for w in query.lower().split() if w not in STOP and len(w) > 2}
    desc_lower  = description.lower()

    if not query_words:
        return 0.5   # nothing meaningful to compare

    hits = sum(1 for w in query_words if w in desc_lower)
    return hits / len(query_words)


def _pick_best(query: str, candidates: list[dict]) -> dict:
    """
    Score each USDA candidate.  Word overlap is PRIMARY (±4 pts).
    Calorie plausibility is SECONDARY (±1-2 pts).

    Full scoring table:
      Word overlap == 1.0  (all query words found) : +4
      Word overlap >= 0.5  (majority found)        : +2
      Word overlap  > 0.0  (some found)            : +1
      Word overlap == 0.0  (zero found)            : -4   ← kills basil/chicken-finger
      Wrong food category in description           : -10  ← hard reject
      kcal in 50-450 range                         : +1   (plausible whole food)
      kcal > 500                                   : -2   (oil/raw grain)
      kcal < 5 AND query NOT a vegetable           : -1   (plain water/broth)
      NOTE: NO penalty for kcal < 20 when query IS a vegetable ← fixes tomato bug
      'cooked' in desc AND query implies cooked    : +2
      kcal > 300 AND query implies cooked          : -1
      Specialty variant in desc not in query       : -3
    """
    q = query.lower()

    cooked_signals = {
        "cooked","boiled","steamed","fried","baked","roasted",
        "rice","dal","pasta","lentil","bean","curry","porridge",
    }
    query_implies_cooked  = any(w in q for w in cooked_signals)
    query_is_vegetable    = any(v in q for v in LOW_CAL_VEGETABLES)

    scored = []
    for c in candidates:
        score   = 0
        kcal    = c.get("calories", 0) or 0
        desc    = c.get("_usda_description", "")
        desc_lc = desc.lower()

        # ── PRIMARY: word overlap ──────────────────────────────
        overlap = _word_overlap(query, desc)
        if   overlap == 1.0: score += 4
        elif overlap >= 0.5: score += 2
        elif overlap  > 0.0: score += 1
        else:                score -= 4   # completely different food

        # ── Hard reject: wrong food category ──────────────────
        for bad in WRONG_CATEGORY:
            if bad in desc_lc and bad not in q:
                score -= 10
                logger.debug(f"[nutrition] Hard-reject '{desc}' — wrong category '{bad}'")
                break

        # ── Calorie plausibility (secondary) ──────────────────
        if 50 <= kcal <= 450:
            score += 1
        if kcal > 500:
            score -= 2

        # Low-calorie fix: only penalise if truly water-like AND not a vegetable
        if kcal < 5 and not query_is_vegetable:
            score -= 1
        # Previously: kcal < 20 → -1  (this made tomato lose to basil — REMOVED)

        # ── Cooking state ──────────────────────────────────────
        if query_implies_cooked and "cooked" in desc_lc:
            score += 2
        if query_implies_cooked and kcal > 300:
            score -= 1

        # ── Specialty variant penalty ──────────────────────────
        for v in SPECIALTY_VARIANTS:
            if v in desc_lc and v not in q:
                score -= 3; break

        scored.append((score, c))
        logger.debug(
            f"[nutrition] {score:+3d} pts | overlap={overlap:.2f} | "
            f"{kcal:5.0f} kcal | '{desc}'"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    best, best_score = scored[0][1], scored[0][0]

    # Warn when best match has zero word overlap — likely still wrong
    if _word_overlap(query, best.get("_usda_description", "")) == 0.0:
        logger.warning(
            f"[nutrition] ⚠ Zero overlap: query='{query}' matched "
            f"'{best.get('_usda_description')}' (score={best_score}). "
            f"Add '{query}' to FOOD_NAME_MAP in normalize.py."
        )
    return best


def lookup_openfoodfacts(name: str) -> dict | None:
    try:
        resp = requests.get(
            "https://world.openfoodfacts.org/cgi/search.pl",
            params={"search_terms": name, "json": 1, "page_size": 1,
                    "fields": "nutriments,product_name"},
            timeout=10,
        )
        resp.raise_for_status()
        products = resp.json().get("products", [])
        if not products: return None
        n = products[0].get("nutriments", {})
        logger.debug(f"[nutrition] OFF: '{products[0].get('product_name')}'")
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
    except Exception as e:
        logger.error(f"[nutrition] OpenFoodFacts error: {e}"); return None


def scale_nutrients(per_100g: dict, grams: int) -> dict:
    f = grams / 100.0
    scaled = {}
    for k, v in per_100g.items():
        if k.startswith("_"):
            scaled[k] = v
        else:
            try:    scaled[k] = round(float(v) * f, 2)
            except: scaled[k] = 0.0
    scaled["_grams"] = grams
    return scaled


def aggregate_meal(items: list[dict]) -> dict:
    totals = {k: 0.0 for k in RDI}
    totals["_grams"] = 0
    for item in items:
        n = item.get("nutrients", {})
        for k in totals:
            totals[k] += float(n.get(k, 0) or 0)
    kcal = totals["calories"] or 1
    totals["_carb_pct"]    = round(totals["carbs_g"]   * 4 / kcal * 100, 1)
    totals["_protein_pct"] = round(totals["protein_g"] * 4 / kcal * 100, 1)
    totals["_fat_pct"]     = round(totals["fat_g"]     * 9 / kcal * 100, 1)
    macro_keys = {"calories","carbs_g","fiber_g","sugar_g","protein_g","fat_g","saturated_fat_g"}
    for k, rdi in RDI.items():
        if k not in macro_keys:
            totals[f"_rdi_{k}"] = round(totals.get(k, 0) / rdi * 100, 1) if rdi else 0
    return totals


def _parse_usda_nutrients(n: dict) -> dict:
    def g(*keys):
        for k in keys:
            if k in n: return float(n[k])
        return 0.0
    return {
        "calories":        g("Energy"),
        "carbs_g":         g("Carbohydrate, by difference"),
        "fiber_g":         g("Fiber, total dietary"),
        "sugar_g":         g("Sugars, total including NLEA"),
        "protein_g":       g("Protein"),
        "fat_g":           g("Total lipid (fat)"),
        "saturated_fat_g": g("Fatty acids, total saturated"),
        "sodium_mg":       g("Sodium, Na"),
        "iron_mg":         g("Iron, Fe"),
        "calcium_mg":      g("Calcium, Ca"),
        "zinc_mg":         g("Zinc, Zn"),
        "magnesium_mg":    g("Magnesium, Mg"),
        "potassium_mg":    g("Potassium, K"),
        "phosphorus_mg":   g("Phosphorus, P"),
        "vit_a_mcg":       g("Vitamin A, RAE"),
        "vit_c_mg":        g("Vitamin C, total ascorbic acid"),
        "vit_d_mcg":       g("Vitamin D (D2 + D3)"),
        "vit_e_mg":        g("Vitamin E (alpha-tocopherol)"),
        "vit_k_mcg":       g("Vitamin K (phylloquinone)"),
        "vit_b1_mg":       g("Thiamin"),
        "vit_b2_mg":       g("Riboflavin"),
        "vit_b3_mg":       g("Niacin"),
        "vit_b6_mg":       g("Vitamin B-6"),
        "vit_b12_mcg":     g("Vitamin B-12"),
    }
