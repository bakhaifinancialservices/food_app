# utils/label_ocr.py — Food label OCR
# Providers: "groq" | "gemini"  (Qwen2 excluded — unreliable for numbers)
# Two-pass strategy:
#   Pass 1: read nutrition table directly
#   Pass 2: if key fields still null, estimate from ingredients list

import os, re, io, json, base64, logging
from groq import Groq
logger = logging.getLogger(__name__)

LABEL_SYSTEM = (
    "You are a nutrition label extraction assistant. "
    "Return ONLY valid JSON — no prose, no markdown."
)

LABEL_PASS1 = """Extract the nutrition facts table from this food label image.

RULES:
1. Read every number carefully — do NOT guess or infer values
2. Convert all values to the units specified below (g, mg, kcal, mcg)
3. If a nutrient is listed as %DV only (no gram value), set it to null
4. If the label is partially visible, extract what you can; set the rest to null
5. Hindi/regional language labels: numbers are still readable — extract them
6. Indian labels often omit micronutrients — that is normal, set those to null

Return ONLY this JSON (use null for any field not on the label):
{
  "product_name"       : "string or null",
  "brand"              : "string or null",
  "serving_size_g"     : number or null,
  "servings_per_package": number or null,
  "per_serving": {
    "calories_kcal"    : number or null,
    "total_fat_g"      : number or null,
    "saturated_fat_g"  : number or null,
    "trans_fat_g"      : number or null,
    "carbohydrates_g"  : number or null,
    "fiber_g"          : number or null,
    "sugar_g"          : number or null,
    "protein_g"        : number or null,
    "sodium_mg"        : number or null,
    "calcium_mg"       : number or null,
    "iron_mg"          : number or null,
    "potassium_mg"     : number or null,
    "vit_a_mcg"        : number or null,
    "vit_c_mg"         : number or null,
    "vit_d_mcg"        : number or null
  },
  "ingredients_text"   : "full ingredients list visible on label, or null",
  "label_quality"      : "clear|angled|partial|low_contrast|unreadable",
  "language"           : "english|hindi|mixed|other",
  "missing_fields_reason": "why key fields are null (not printed, not visible, cut off)"
}"""

PASS2_TEMPLATE = """A food product has these known nutrition values per serving:
{known}

Its ingredients list reads:
{ingredients}

Estimate likely values for these missing fields: {missing}

Rules:
- Only estimate when ingredients provide strong evidence (e.g. 'whole wheat' → fiber)
- Be conservative — if unsure, return null
- Do NOT invent values

Return ONLY a JSON object with the missing field names as keys.
Example: {{"fiber_g": 2.5, "calcium_mg": null}}"""


def extract_label(image_bytes: bytes, mime_type: str = "image/jpeg",
                  provider: str = "groq") -> dict:
    """
    Two-pass label extraction.
    provider: 'groq' | 'gemini'  (Qwen2 not supported for labels)
    """
    b64      = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime_type};base64,{b64}"

    logger.info(f"[label_ocr] Pass 1 — provider='{provider}'")

    if provider == "gemini":
        try:
            result = _gemini_extract(b64, mime_type)
        except Exception as e:
            logger.warning(f"[label_ocr] Gemini failed: {e} → Groq fallback")
            result = _groq_extract(data_url)
    else:
        result = _groq_extract(data_url)

    if "error" in result:
        return result

    logger.info(
        f"[label_ocr] Pass 1 done: quality={result.get('label_quality')}, "
        f"product={result.get('product_name')}"
    )

    # Pass 2: estimate missing fields from ingredients
    per_serving  = result.get("per_serving", {})
    ingredients  = result.get("ingredients_text") or ""
    estimatable  = ["fiber_g","saturated_fat_g","sugar_g","sodium_mg",
                    "calcium_mg","iron_mg","vit_c_mg"]
    missing      = [f for f in estimatable if per_serving.get(f) is None]

    if missing and ingredients and len(missing) >= 2:
        logger.info(f"[label_ocr] Pass 2: estimating {missing} from ingredients")
        estimates = _estimate_from_ingredients(per_serving, ingredients, missing)
        for field, value in estimates.items():
            if field in per_serving and per_serving[field] is None and value is not None:
                per_serving[field] = value
                result.setdefault("_pass2_estimated", []).append(field)
        logger.info(f"[label_ocr] Pass 2 estimated: {result.get('_pass2_estimated', [])}")

    result["per_serving"] = per_serving
    return result


def serving_calculator(per_serving: dict, servings: float) -> dict:
    """Scale per-serving nutrients by number of servings consumed."""
    if not per_serving:
        return {}
    scaled = {}
    for k, v in per_serving.items():
        if v is None:
            scaled[k] = None
        else:
            try:    scaled[k] = round(float(v) * servings, 2)
            except: scaled[k] = None
    return {
        "calories":          scaled.get("calories_kcal"),
        "fat_g":             scaled.get("total_fat_g"),
        "saturated_fat_g":   scaled.get("saturated_fat_g"),
        "carbs_g":           scaled.get("carbohydrates_g"),
        "fiber_g":           scaled.get("fiber_g"),
        "sugar_g":           scaled.get("sugar_g"),
        "protein_g":         scaled.get("protein_g"),
        "sodium_mg":         scaled.get("sodium_mg"),
        "calcium_mg":        scaled.get("calcium_mg"),
        "iron_mg":           scaled.get("iron_mg"),
        "potassium_mg":      scaled.get("potassium_mg"),
        "vit_a_mcg":         scaled.get("vit_a_mcg"),
        "vit_c_mg":          scaled.get("vit_c_mg"),
        "vit_d_mcg":         scaled.get("vit_d_mcg"),
        "_servings_consumed": servings,
        "_is_label_mode":    True,
    }


def _groq_extract(data_url: str) -> dict:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        r = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": LABEL_SYSTEM},
                {"role": "user",   "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text",      "text": LABEL_PASS1},
                ]},
            ],
            max_tokens=1500, temperature=0,
        )
        raw = r.choices[0].message.content.strip()
        logger.debug(f"[label_ocr:groq] {raw[:300]}")
        return _parse(raw)
    except Exception as e:
        logger.error(f"[label_ocr] Groq failed: {e}")
        return {"error": str(e), "per_serving": {}}


def _gemini_extract(b64: str, mime_type: str) -> dict:
    try:
        import google.generativeai as genai
        import PIL.Image
    except ImportError:
        raise RuntimeError("Run: pip install google-generativeai pillow")

    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    img   = PIL.Image.open(io.BytesIO(base64.b64decode(b64)))
    r     = model.generate_content([img, LABEL_SYSTEM + "\n\n" + LABEL_PASS1])
    raw   = r.text.strip()
    logger.debug(f"[label_ocr:gemini] {raw[:300]}")
    return _parse(raw)


def _estimate_from_ingredients(known: dict, ingredients: str, missing: list) -> dict:
    known_str   = ", ".join(f"{k}: {v}" for k, v in known.items() if v is not None)
    missing_str = ", ".join(missing)
    prompt = PASS2_TEMPLATE.format(
        known=known_str or "unknown",
        ingredients=ingredients[:800],
        missing=missing_str,
    )
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=256, temperature=0,
        )
        raw  = r.choices[0].message.content.strip()
        data = json.loads(raw)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except Exception as e:
        logger.warning(f"[label_ocr] Pass 2 estimation failed: {e}")
        return {}


def _parse(raw: str) -> dict:
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and "per_serving" in d:
            return d
    except json.JSONDecodeError:
        pass
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            d = json.loads(m.group())
            if isinstance(d, dict):
                d.setdefault("per_serving", {})
                return d
    except json.JSONDecodeError:
        pass
    logger.error("[label_ocr] Could not parse label response")
    return {"error": "Could not read label. Try a flatter, better-lit photo.",
            "per_serving": {}, "label_quality": "unreadable", "notes": raw[:300]}
