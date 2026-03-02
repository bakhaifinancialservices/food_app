# utils/normalize.py
# ─────────────────────────────────────────────────────────────
# Converts colloquial food names to DB-matchable formal names.
# Primary: static dictionary (100+ Indian + common foods)
# Fallback: Groq Llama 3.3 70B
# ─────────────────────────────────────────────────────────────

import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# ── Static mapping dictionary ─────────────────────────────────
# key: colloquial/regional name (lowercase)
# value: standard nutritional DB name
FOOD_NAME_MAP = {
    # ── Dal / Lentils ─────────────────────────────────────────
    "dal":              "lentils cooked",
    "daal":             "lentils cooked",
    "dal tadka":        "lentils cooked",
    "dal fry":          "lentils cooked",
    "dal makhani":      "lentils cooked",
    "moong dal":        "mung beans cooked",
    "masoor dal":       "red lentils cooked",
    "chana dal":        "chickpeas split cooked",
    "toor dal":         "pigeon peas cooked",
    "urad dal":         "black gram cooked",

    # ── Bread / Roti ─────────────────────────────────────────
    "roti":             "chapati whole wheat",
    "chapati":          "chapati whole wheat",
    "paratha":          "paratha whole wheat",
    "naan":             "naan bread",
    "puri":             "puri fried bread",
    "bhatura":          "bhatura fried bread",
    "kulcha":           "naan bread",

    # ── Rice ─────────────────────────────────────────────────
    "rice":             "rice white long-grain cooked",
    "basmati rice":     "rice basmati cooked",
    "brown rice":       "rice brown long-grain cooked",
    "biryani":          "biryani rice dish",
    "pulao":            "rice pulao cooked",
    "khichdi":          "rice lentil khichdi",
    "fried rice":       "rice fried",

    # ── Curries / Sabzi ──────────────────────────────────────
    "chole":            "chickpeas cooked",
    "chana masala":     "chickpeas cooked",
    "rajma":            "kidney beans cooked",
    "palak paneer":     "spinach cottage cheese curry",
    "paneer butter masala": "cottage cheese curry",
    "matar paneer":     "cottage cheese peas curry",
    "aloo gobi":        "potato cauliflower curry",
    "aloo matar":       "potato peas curry",
    "bhindi":           "okra cooked",
    "baingan":          "eggplant cooked",
    "gobi":             "cauliflower cooked",

    # ── Snacks / Street food ──────────────────────────────────
    "samosa":           "samosa fried pastry",
    "kachori":          "kachori fried pastry",
    "pakora":           "pakora fried fritter",
    "bhajiya":          "pakora fried fritter",
    "vada":             "vada fried lentil",
    "medu vada":        "vada fried lentil",
    "aloo tikki":       "potato patty fried",
    "dhokla":           "dhokla steamed",
    "poha":             "flattened rice poha",
    "upma":             "semolina porridge upma",
    "idli":             "idli steamed rice cake",
    "dosa":             "dosa rice crepe",
    "uttapam":          "uttapam rice pancake",

    # ── Dairy ────────────────────────────────────────────────
    "paneer":           "cottage cheese",
    "curd":             "yogurt plain",
    "dahi":             "yogurt plain",
    "raita":            "yogurt raita",
    "lassi":            "lassi yogurt drink",
    "buttermilk":       "buttermilk",
    "ghee":             "ghee clarified butter",

    # ── Desserts ─────────────────────────────────────────────
    "gulab jamun":      "gulab jamun sweet",
    "rasgulla":         "rasgulla sweet",
    "kheer":            "rice pudding kheer",
    "halwa":            "halwa semolina sweet",
    "ladoo":            "ladoo sweet ball",
    "barfi":            "barfi milk sweet",
    "jalebi":           "jalebi fried sweet",

    # ── Drinks ───────────────────────────────────────────────
    "chai":             "tea with milk",
    "masala chai":      "tea with milk spiced",
    "coffee":           "coffee with milk",
    "nimbu pani":       "lemonade",
    "coconut water":    "coconut water",

    # ── Okra / Lady finger (common confusion with ladyfinger cookie) ──
    "okra":             "okra raw",
    "okra raw":         "okra raw",
    "okra cooked":      "okra cooked",
    "lady finger":      "okra raw",      # Indian English name
    "ladyfinger":       "okra raw",      # same — NOT the dessert cookie
    "ladies finger":    "okra raw",
    "bhindi":           "okra cooked",
    "bhindi masala":    "okra cooked",

    # ── Raw salad vegetables ──────────────────────────────────
    "tomato":           "tomatoes raw",
    "tomato raw":       "tomatoes raw",
    "cherry tomato":    "tomatoes cherry raw",
    "cucumber":         "cucumber raw",
    "cucumber raw":     "cucumber raw",
    "capsicum":         "peppers sweet green raw",
    "green capsicum":   "peppers sweet green raw",
    "red capsicum":     "peppers sweet red raw",
    "bell pepper":      "peppers sweet green raw",
    "onion":            "onions raw",
    "red onion":        "onions red raw",
    "spring onion":     "onions spring raw",
    "lettuce":          "lettuce raw",
    "spinach":          "spinach raw",
    "carrot":           "carrots raw",
    "beetroot":         "beets raw",
    "radish":           "radishes raw",
    "corn":             "corn sweet cooked",
    "sweet corn":       "corn sweet cooked",
    "mixed salad":      "salad mixed greens",

    # ── Common international foods ────────────────────────────
    "pasta":            "pasta cooked",
    "pizza":            "pizza",
    "burger":           "hamburger",
    "sandwich":         "sandwich",
    "salad":            "mixed salad",
    "soup":             "vegetable soup",
    "omelette":         "omelette egg",
    "fried egg":        "egg fried",
    "boiled egg":       "egg hard boiled",
    "chicken":          "chicken cooked",
    "fish":             "fish cooked",
}

# ── Vague terms that should trigger disambiguation ────────────
VAGUE_TERMS = {
    "curry", "food", "dish", "meal", "item", "stuff", "sauce",
    "thing", "indian food", "rice dish", "bread", "vegetable",
    "fruit", "meat", "snack", "dessert", "drink", "beverage",
    "mixed", "unknown", "various",
}


def normalize_name(food_name: str) -> str:
    """
    Return a DB-matchable food name.
    Logs which path was used (dict / Groq).
    """
    name_lower = food_name.lower().strip()

    # 1. Exact dict match
    if name_lower in FOOD_NAME_MAP:
        normalized = FOOD_NAME_MAP[name_lower]
        logger.debug(f"[normalize] '{food_name}' → '{normalized}' (dict exact)")
        return normalized

    # 2. Partial dict match (food name contains a key)
    for key, value in FOOD_NAME_MAP.items():
        if key in name_lower:
            logger.debug(f"[normalize] '{food_name}' → '{value}' (dict partial: '{key}')")
            return value

    # 3. Groq fallback
    logger.info(f"[normalize] No dict match for '{food_name}' — asking Groq")
    return _groq_normalize(food_name)


def is_vague(food_name: str) -> bool:
    """Return True if food_name is too generic to look up reliably."""
    return food_name.lower().strip() in VAGUE_TERMS


def _groq_normalize(food_name: str) -> str:
    """Ask Groq Llama 3.3 70B for the standard DB name."""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a nutritional database assistant. Return only the standard food database name, no explanation.",
                },
                {
                    "role": "user",
                    "content": (
                        f"What is the standard USDA nutritional database search term for: '{food_name}' "
                        "as typically eaten in an Indian meal context? "
                        "IMPORTANT: 'lady finger' and 'ladyfinger' means OKRA (the vegetable), NOT the cookie. "
                        "'tomato' means the red fruit/vegetable, NOT basil or any herb. "
                        "Reply with just the food name, 1-5 words only."
                    ),
                },
            ],
            max_tokens=20,
            temperature=0,
        )
        normalized = response.choices[0].message.content.strip().lower()
        logger.info(f"[normalize] Groq mapped '{food_name}' → '{normalized}'")
        return normalized
    except Exception as e:
        logger.error(f"[normalize] Groq normalization failed for '{food_name}': {e} — using original name")
        return food_name  # safe fallback: use original
