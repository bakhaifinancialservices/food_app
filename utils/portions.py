# utils/portions.py — Convert portion descriptions to grams.
# Uses: utensil capacity × fill level → grams (most accurate)
#   then: per-food dict lookup
#   then: generic unit fallback
#   then: Groq estimate (last resort)

import os, logging
from groq import Groq
logger = logging.getLogger(__name__)

UTENSIL_ML = {
    "small katori": 110, "medium katori": 170, "large katori": 230,
    "small bowl":   180, "medium bowl":   320, "large bowl":   450,
    "plate":        700, "small plate":   350, "side plate":   280,
    "teacup":       150, "glass":         240, "mug":          300,
}
FILL_MULT = {"25%": 0.25, "50%": 0.50, "75%": 0.75, "full": 1.00, "n/a": 0.70}

PORTION_TO_GRAMS = {
    # Indian breads
    ("roti","piece"):40,("chapati","piece"):40,("paratha","piece"):80,
    ("naan","piece"):90,("puri","piece"):30,("bhatura","piece"):80,
    # Rice
    ("rice","cup"):186,("rice","bowl"):200,("biryani","bowl"):280,
    ("pulao","cup"):190,("fried rice","cup"):200,
    # South Indian
    ("idli","piece"):40,("dosa","piece"):80,("vada","piece"):50,("uttapam","piece"):90,
    # Dal / curry
    ("dal","bowl"):250,("dal","cup"):200,("curry","bowl"):250,
    ("sabzi","bowl"):200,("chole","bowl"):250,("rajma","bowl"):250,("sambar","bowl"):200,
    # Snacks
    ("samosa","piece"):60,("kachori","piece"):55,("pakora","piece"):30,
    ("dhokla","piece"):35,("bread","slice"):30,("biscuit","piece"):10,
    # Breakfast
    ("poha","bowl"):180,("upma","bowl"):200,("oats","cup"):240,
    # Raw vegetables — per piece
    ("tomato","piece"):120,("cucumber","piece"):150,("onion","piece"):100,
    ("capsicum","piece"):120,("carrot","piece"):80,("okra","piece"):6,
    # Raw vegetables — per gram (model outputs grams directly)
    ("tomato","grams"):1,("cucumber","grams"):1,("onion","grams"):1,
    ("capsicum","grams"):1,("okra","grams"):1,("spinach","grams"):1,
    # Fruits
    ("apple","piece"):182,("banana","piece"):118,("orange","piece"):131,
    ("mango","piece"):200,("grapes","cup"):150,
    # Proteins
    ("egg","piece"):50,("chicken","piece"):120,("fish","piece"):120,
    ("paneer","cup"):226,("paneer","piece"):50,
    # Generic fallbacks
    ("*","cup"):240,("*","bowl"):250,("*","tablespoon"):15,
    ("*","teaspoon"):5,("*","slice"):30,("*","piece"):80,("*","grams"):1,
}


def to_grams(food_name: str, portion_size: float, portion_unit: str,
             container_type: str = "", fill_level: str = "n/a") -> int:
    food  = food_name.lower().strip()
    unit  = portion_unit.lower().strip()
    ctype = container_type.lower().strip()

    # 1. Direct grams
    if unit in ("grams", "g", "gram"):
        g = max(1, int(round(portion_size)))
        logger.debug(f"[portions] '{food_name}' → {g}g (direct)")
        return g

    # 2. Utensil capacity × fill level
    if ctype and ctype not in ("unknown", "none", ""):
        cap = _utensil_ml(ctype)
        if cap:
            fill = FILL_MULT.get(fill_level, 0.70)
            g = max(5, int(round(cap * fill * 0.85 * portion_size)))
            logger.debug(
                f"[portions] '{food_name}' utensil='{ctype}' "
                f"cap={cap}ml fill={fill_level} → {g}g"
            )
            return g

    # 3. Specific food dict
    for (fkey, ukey), gval in PORTION_TO_GRAMS.items():
        if fkey == "*": continue
        if fkey in food and ukey == unit:
            g = int(round(portion_size * gval))
            logger.debug(f"[portions] '{food_name}' dict '{fkey}' → {g}g")
            return g

    # 4. Generic unit
    if ("*", unit) in PORTION_TO_GRAMS:
        g = int(round(portion_size * PORTION_TO_GRAMS[("*", unit)]))
        logger.debug(f"[portions] '{food_name}' generic unit → {g}g")
        return g

    # 5. Groq last resort
    logger.warning(f"[portions] No match for '{food_name}' {portion_size} {unit} → Groq")
    return _groq_estimate(food_name, portion_size, unit)


def _utensil_ml(ctype: str) -> int:
    for k, v in UTENSIL_ML.items():
        if k in ctype: return v
    if "small"  in ctype: return 150
    if "medium" in ctype: return 250
    if "large"  in ctype: return 400
    if "bowl"   in ctype: return 300
    if "plate"  in ctype: return 500
    if "cup"    in ctype: return 200
    if "katori" in ctype: return 170
    return 0


def _groq_estimate(food: str, size: float, unit: str) -> int:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content":
                f"Weight in grams: {size} {unit} of {food}. Reply with a single integer only."}],
            max_tokens=10, temperature=0,
        )
        raw = r.choices[0].message.content.strip()
        g   = int("".join(filter(str.isdigit, raw)) or "100")
        logger.info(f"[portions] Groq → {g}g for '{food}'")
        return g
    except Exception as e:
        logger.error(f"[portions] Groq failed: {e} → 100g default")
        return 100
