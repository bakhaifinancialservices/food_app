# utils/vision.py — Food detection from images.
#
# Meal photo providers:
#   "groq"   → Groq Llama 4 Scout 17B  (fast, free, default)
#   "gemini" → Google Gemini 1.5 Flash  (accurate, free tier)
#   "qwen2"  → Qwen2-VL 7B via HuggingFace (needs HF_TOKEN, 10-30s)
#
# Food label providers: "groq" | "gemini" only
#   (Qwen2 excluded — unreliable for precise number extraction)
#
# Fallback: selected provider → groq → raise

import os, re, io, json, base64, logging
from groq import Groq
logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────
UTENSIL_CONTEXT = """
UTENSIL SIZE REFERENCE:
- Small katori/bowl : 100-120 ml → 80-100 g food
- Medium katori/bowl: 150-200 ml → 120-160 g food
- Large bowl        : 300-400 ml → 250-320 g food
- Standard plate    : estimate each item separately
- Tablespoon: ~15 g | Teaspoon: ~5 g
Per-piece weights:
- 1 roti/chapati: 40g | 1 paratha: 80g | 1 naan: 90g | 1 idli: 40g | 1 dosa: 80g
- 1 medium tomato: 120g | 1 medium onion: 100g | 1 medium cucumber: 150g | 1 capsicum: 120g
Steps: (1) identify container, (2) estimate fill %, (3) capacity × fill × 0.85 = grams
"""

SYSTEM_PROMPT = (
    "You are a precise food detection assistant specialising in Indian and international cuisine. "
    "Return ONLY valid JSON — no prose, no markdown, no explanation."
)

USER_PROMPT = """Examine this image and list every food item visible.

RULES:
1. Be SPECIFIC: 'tomato raw' not 'vegetable', 'okra' not 'green vegetable', 'white rice' not 'food'
2. Raw salad: one JSON object per vegetable, not one object for the whole salad
3. Thali/mixed plate: one object per item/katori
4. Use UTENSIL CONTEXT below to estimate grams — container first, then fill level
5. confidence >= 0.85 ONLY when certain of BOTH identity AND portion size
6. NEVER label okra/lady finger as 'finger food' or any non-vegetable
7. NEVER label tomato as 'basil' or any herb

""" + UTENSIL_CONTEXT + """

Each JSON object must have EXACTLY these fields:
  food_name         : "specific name e.g. tomato raw, okra cooked, white rice"
  container_type    : "small katori|medium katori|large bowl|plate|piece|none"
  fill_level        : "25%|50%|75%|full|n/a"
  portion_size      : <float>
  portion_unit      : "piece|cup|bowl|grams|slice|tablespoon"
  confidence        : <float 0-1>
  visual_description: "one sentence"

Example:
[
  {"food_name":"tomato raw","container_type":"medium bowl","fill_level":"n/a","portion_size":100,"portion_unit":"grams","confidence":0.88,"visual_description":"Diced red tomato pieces in a salad bowl."},
  {"food_name":"cucumber raw","container_type":"medium bowl","fill_level":"n/a","portion_size":75,"portion_unit":"grams","confidence":0.90,"visual_description":"Sliced cucumber in the same bowl."}
]

If no food visible return: []
Return ONLY the JSON array."""


def detect_foods(image_bytes: bytes, mime_type: str = "image/jpeg",
                 provider: str = "groq") -> list[dict]:
    """Detect food items. provider: 'groq'|'gemini'|'qwen2'"""
    b64      = base64.b64encode(image_bytes).decode()
    data_url = f"data:{mime_type};base64,{b64}"
    logger.info(f"[vision] provider='{provider}'")

    if provider == "gemini":
        try:
            return _gemini(b64, mime_type)
        except Exception as e:
            logger.warning(f"[vision] Gemini failed: {e} → Groq fallback")

    elif provider == "qwen2":
        try:
            return _qwen2(b64, mime_type)
        except Exception as e:
            logger.warning(f"[vision] Qwen2-VL failed: {e} → Groq fallback")

    return _groq(data_url)   # default / fallback


def _groq(data_url: str) -> list[dict]:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    r = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",      "text": USER_PROMPT},
            ]},
        ],
        max_tokens=2048, temperature=0.1,
    )
    raw = r.choices[0].message.content.strip()
    logger.debug(f"[vision:groq] {raw[:200]}")
    return _parse(raw)


def _gemini(b64: str, mime_type: str) -> list[dict]:
    try:
        import google.generativeai as genai
        import PIL.Image
    except ImportError:
        raise RuntimeError("Run: pip install google-generativeai pillow")

    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    genai.configure(api_key=key)
    model  = genai.GenerativeModel("gemini-1.5-flash")
    img    = PIL.Image.open(io.BytesIO(base64.b64decode(b64)))
    r      = model.generate_content([img, SYSTEM_PROMPT + "\n\n" + USER_PROMPT])
    raw    = r.text.strip()
    logger.debug(f"[vision:gemini] {raw[:200]}")
    return _parse(raw)


def _qwen2(b64: str, mime_type: str) -> list[dict]:
    """Qwen2-VL 7B via HuggingFace Serverless Inference API. Needs HF_TOKEN."""
    import requests as rq

    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not set in .env — get a free token at huggingface.co/settings/tokens"
        )

    resp = rq.post(
        "https://api-inference.huggingface.co/models/Qwen/Qwen2-VL-7B-Instruct",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "inputs": {
                "image": f"data:{mime_type};base64,{b64}",
                "text":  SYSTEM_PROMPT + "\n\n" + USER_PROMPT,
            },
            "parameters": {"max_new_tokens": 2048},
        },
        timeout=90,
    )

    if resp.status_code == 503:
        raise RuntimeError("Qwen2-VL is loading (cold start). Wait ~20s and retry.")
    if resp.status_code != 200:
        raise RuntimeError(f"HuggingFace {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    text = (data[0].get("generated_text", "") if isinstance(data, list)
            else data.get("generated_text", str(data)))
    logger.debug(f"[vision:qwen2] {text[:200]}")
    return _parse(text)


def _parse(raw: str) -> list[dict]:
    """3-layer JSON parsing: direct → regex → Groq reformat."""
    # Layer 1
    try:
        d = json.loads(raw)
        if isinstance(d, list): return _validate(d)
    except json.JSONDecodeError:
        pass
    # Layer 2
    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            d = json.loads(m.group())
            if isinstance(d, list): return _validate(d)
    except json.JSONDecodeError:
        pass
    # Layer 3: Groq reformat
    logger.warning("[vision] Bad JSON — Groq reformat attempt")
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        fix = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Convert to valid JSON array of food objects. Return ONLY the array."},
                {"role": "user",   "content": raw[:3000]},
            ],
            max_tokens=2048, temperature=0,
        )
        d = json.loads(fix.choices[0].message.content.strip())
        if isinstance(d, list): return _validate(d)
    except Exception as e:
        logger.error(f"[vision] All JSON parsing failed: {e}")
    return []


def _validate(items: list) -> list[dict]:
    valid_units = {"piece", "cup", "bowl", "grams", "slice", "tablespoon"}
    out = []
    for item in items:
        if not isinstance(item, dict): continue
        item.setdefault("food_name",          "unknown food")
        item.setdefault("container_type",     "unknown")
        item.setdefault("fill_level",         "n/a")
        item.setdefault("portion_size",       1.0)
        item.setdefault("portion_unit",       "piece")
        item.setdefault("confidence",         0.5)
        item.setdefault("visual_description", "")
        item["confidence"]   = max(0.0, min(1.0, float(item["confidence"])))
        item["portion_size"] = max(0.01, float(item["portion_size"]))
        if item["portion_unit"] not in valid_units:
            item["portion_unit"] = "piece"
        logger.debug(
            f"[vision] ✓ {item['food_name']} | "
            f"{item['portion_size']} {item['portion_unit']} | "
            f"{item['confidence']:.0%} | {item['container_type']} {item['fill_level']}"
        )
        out.append(item)
    return out
