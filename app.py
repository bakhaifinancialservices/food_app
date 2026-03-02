# app.py — NutriScan AI
# streamlit run app.py
#
# Meal photo flow (3 stages):
#   Stage 1 — Upload photo → click "Analyze Meal"
#   Stage 2 — Review & correct every detected item (name + qty + unit) → "Confirm All"
#   Stage 3 — Full nutrition analysis results
#
# Label flow (unchanged — 2 stages):
#   Stage 1 — Upload label photo → click "Extract Label Info"
#   Stage 2 — Servings selector + nutrition results

import logging, os
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from utils.vision    import detect_foods
from utils.portions  import to_grams
from utils.normalize import normalize_name
from utils.nutrition import lookup_nutrition, scale_nutrients, aggregate_meal, RDI
from utils.validate  import run_validation, nutrition_sanity_check
from utils.label_ocr import extract_label, serving_calculator

FEATURE_FLAGS = {
    "meal_history":False,"barcode_scan":False,"rdi_dashboard":False,
    "pdf_export":False,"calorie_goals":False,"api_endpoint":False,
    "multi_language":False,"diet_tagging":False,
}

PORTION_UNITS = ["grams","piece","cup","bowl","tablespoon","slice"]

st.set_page_config(
    page_title="NutriScan AI", page_icon="🥗",
    layout="wide", initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        # Stage tracking
        "meal_stage":        1,     # 1=upload, 2=review, 3=results
        # Detection output (Stage 2)
        "detected_items":    [],    # raw from vision + validation, before nutrition
        # Confirmed output (Stage 3)
        "meal_items":        [],    # after nutrition fetch
        "meal_total":        {},
        "sanity_warnings":   [],
        # Image storage (for Re-analyse)
        "meal_image_bytes":  None,
        "meal_image_mime":   "image/jpeg",
        # Label
        "label_data":        {},
        "label_total":       {},
        "label_done":        False,
        # Settings
        "debug_mode":        False,
        "meal_provider":     "groq",
        "label_provider":    "groq",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _mime(f):
    n = getattr(f, "name", "x.jpg").lower()
    return ("image/png"  if n.endswith(".png")  else
            "image/webp" if n.endswith(".webp") else "image/jpeg")

def _read(f):
    f.seek(0); return f.read()

def _fetch_nutrition(food_name, portion_size, portion_unit, container="", fill="n/a"):
    grams      = to_grams(food_name, portion_size, portion_unit, container, fill)
    normalized = normalize_name(food_name)
    raw        = lookup_nutrition(normalized)
    if raw:
        return (scale_nutrients(raw, grams), grams, normalized,
                raw.get("_source","?"), raw.get("_usda_description",""))
    return {}, grams, normalized, "none", ""

def _reset_meal():
    st.session_state.meal_stage       = 1
    st.session_state.detected_items   = []
    st.session_state.meal_items       = []
    st.session_state.meal_total       = {}
    st.session_state.sanity_warnings  = []
    st.session_state.meal_image_bytes = None


# ─────────────────────────────────────────────────────────────
# STAGE 1 → 2  :  detect + validate, store, go to review
# ─────────────────────────────────────────────────────────────
def run_detection(image_bytes: bytes, mime: str):
    """Detect foods and run validation. Does NOT fetch nutrition yet."""
    provider = st.session_state.meal_provider

    with st.spinner(f"🔍 Detecting food items  [{provider.upper()}]…"):
        try:
            detected = detect_foods(image_bytes, mime, provider=provider)
            logger.info(f"[app] Detected {len(detected)} item(s)")
        except Exception as e:
            st.error(f"❌ Detection failed: {e}")
            logger.error("[app] Detection error", exc_info=True)
            return

    if not detected:
        st.warning("⚠️ No food items detected — try a clearer photo with better lighting.")
        return

    with st.spinner("🧠 Validating detections…"):
        try:
            enriched, _ = run_validation(detected)
        except Exception as e:
            st.warning(f"⚠️ Validation issue: {e}")
            enriched = detected
            logger.error("[app] Validation error", exc_info=True)

    st.session_state.detected_items = enriched
    st.session_state.meal_stage     = 2   # → Review screen
    logger.info(f"[app] Moving to review stage with {len(enriched)} item(s)")


# ─────────────────────────────────────────────────────────────
# STAGE 2 → 3  :  fetch nutrition with user-confirmed values
# ─────────────────────────────────────────────────────────────
def run_nutrition_from_confirmed(confirmed_items: list[dict]):
    """
    confirmed_items: list of dicts each with:
        food_name, portion_size, portion_unit,
        container_type, fill_level (original vision values kept)
    """
    with st.spinner("📊 Fetching nutrition for confirmed items…"):
        enriched = []
        for item in confirmed_items:
            try:
                nutrients, grams, norm, source, usda_match = _fetch_nutrition(
                    item["food_name"],
                    item["portion_size"],
                    item["portion_unit"],
                    item.get("container_type", ""),
                    item.get("fill_level", "n/a"),
                )
                item["nutrients"]        = nutrients
                item["_grams"]           = grams
                item["_normalized_name"] = norm
                item["_data_source"]     = source
                item["_usda_match"]      = usda_match
                item["_no_data"]         = not bool(nutrients)
            except Exception as e:
                item["nutrients"] = {}
                item["_error"]    = str(e)
                logger.error(f"[app] Nutrition error for '{item['food_name']}'", exc_info=True)
            enriched.append(item)

    try:
        meal_total = aggregate_meal(enriched)
        warnings   = nutrition_sanity_check(enriched, meal_total)
        st.session_state.meal_items      = enriched
        st.session_state.meal_total      = meal_total
        st.session_state.sanity_warnings = warnings
        st.session_state.meal_stage      = 3   # → Results
        logger.info(f"[app] Nutrition done — {meal_total.get('calories',0):.0f} kcal total")
    except Exception as e:
        st.error(f"❌ Aggregation failed: {e}")
        logger.error("[app] Aggregation error", exc_info=True)


# ─────────────────────────────────────────────────────────────
# LABEL PIPELINE  (unchanged)
# ─────────────────────────────────────────────────────────────
def run_label_extraction(image_bytes: bytes, mime: str):
    st.session_state.label_done  = False
    st.session_state.label_data  = {}
    st.session_state.label_total = {}
    provider = st.session_state.label_provider
    with st.spinner(f"📖 Reading label  [{provider.upper()}]…"):
        try:
            data = extract_label(image_bytes, mime, provider=provider)
            st.session_state.label_data = data
            st.session_state.label_done = True
        except Exception as e:
            st.error(f"❌ Label extraction failed: {e}")
            logger.error("[app] Label error", exc_info=True)


# ─────────────────────────────────────────────────────────────
# RENDER — STAGE 2: REVIEW & CONFIRM SCREEN
# ─────────────────────────────────────────────────────────────
def render_review_screen():
    """
    Show every detected item as an editable row.
    User can change: food name (text), quantity (number), unit (selectbox).
    Flags and confidence shown as reference — not blocking.
    Single 'Confirm All & Get Nutrition' button at the bottom.
    Also shows an 'Add Item' button to add items the model missed.
    """
    items = st.session_state.detected_items

    st.subheader("✏️ Review Detected Items")
    st.caption(
        "The AI has detected the items below. "
        "Edit any name or quantity before confirming — then click **Confirm All**."
    )

    # Show the meal image small for reference
    if st.session_state.meal_image_bytes:
        c_img, c_tbl = st.columns([1, 3])
        with c_img:
            st.image(st.session_state.meal_image_bytes,
                     caption="Your meal", use_container_width=True)
    else:
        c_tbl = st.container()

    with c_tbl:
        st.divider()

        # ── Column headers ────────────────────────────────────
        h0, h1, h2, h3, h4, h5 = st.columns([0.3, 3, 1.2, 1.5, 1, 1])
        h0.caption("🤖")
        h1.caption("**Food Name**")
        h2.caption("**Quantity**")
        h3.caption("**Unit**")
        h4.caption("**Confidence**")
        h5.caption("**Container**")
        st.divider()

        # ── One editable row per item ─────────────────────────
        confirmed = []
        for i, item in enumerate(items):
            conf       = item.get("confidence", 0)
            flag       = item.get("is_flagged", False)
            needs_dis  = item.get("needs_disambiguation", False)
            container  = item.get("container_type", "")
            fill       = item.get("fill_level", "n/a")
            orig_unit  = item.get("portion_unit", "piece")
            orig_size  = item.get("portion_size", 1.0)
            flag_reasons = item.get("flag_reasons", [])

            # Row indicator
            c0, c1, c2, c3, c4, c5 = st.columns([0.3, 3, 1.2, 1.5, 1, 1])

            # Confidence badge icon
            with c0:
                icon = "🟢" if conf >= 0.85 else ("🟡" if conf >= 0.65 else "🔴")
                st.markdown(f"<br>{icon}", unsafe_allow_html=True)

            # Editable food name
            with c1:
                new_name = st.text_input(
                    f"name_{i}",
                    value=item["food_name"],
                    key=f"review_name_{i}",
                    label_visibility="collapsed",
                    placeholder="Food name…",
                )

            # Editable quantity
            with c2:
                new_qty = st.number_input(
                    f"qty_{i}",
                    value=float(orig_size),
                    min_value=0.1,
                    step=0.5,
                    key=f"review_qty_{i}",
                    label_visibility="collapsed",
                )

            # Unit selector
            with c3:
                unit_idx = PORTION_UNITS.index(orig_unit) if orig_unit in PORTION_UNITS else 0
                new_unit = st.selectbox(
                    f"unit_{i}",
                    options=PORTION_UNITS,
                    index=unit_idx,
                    key=f"review_unit_{i}",
                    label_visibility="collapsed",
                )

            # Confidence display
            with c4:
                st.markdown(f"<br><span style='font-size:0.85em'>{conf:.0%}</span>",
                            unsafe_allow_html=True)

            # Container info
            with c5:
                cdisp = container if container not in ("", "unknown", "none") else "—"
                if fill and fill != "n/a":
                    cdisp += f" {fill}"
                st.markdown(f"<br><span style='font-size:0.8em;color:grey'>{cdisp}</span>",
                            unsafe_allow_html=True)

            # Flag reasons shown inline (not blocking)
            if flag_reasons:
                for reason in flag_reasons:
                    st.caption(f"  ⚠️ _{reason}_")

            # Build confirmed item — carry original container/fill for gram calc
            confirmed.append({
                **item,                        # keep all original fields
                "food_name":    new_name.strip() or item["food_name"],
                "portion_size": new_qty,
                "portion_unit": new_unit,
            })

        st.divider()

        # ── Add missing item row ──────────────────────────────
        with st.expander("➕ Add an item the AI missed"):
            ac1, ac2, ac3, ac4 = st.columns([3, 1.2, 1.5, 1])
            with ac1:
                extra_name = st.text_input("Food name", key="extra_name",
                                           placeholder="e.g. raita, pickle, papad")
            with ac2:
                extra_qty  = st.number_input("Qty", value=1.0, min_value=0.1,
                                             step=0.5, key="extra_qty")
            with ac3:
                extra_unit = st.selectbox("Unit", PORTION_UNITS, key="extra_unit")
            with ac4:
                st.write("")
                st.write("")
                if st.button("Add", key="add_extra"):
                    if extra_name.strip():
                        new_item = {
                            "food_name":    extra_name.strip(),
                            "portion_size": extra_qty,
                            "portion_unit": extra_unit,
                            "confidence":   1.0,    # user-added = fully trusted
                            "container_type": "",
                            "fill_level":   "n/a",
                            "visual_description": "Added manually by user",
                            "is_flagged":   False,
                            "needs_disambiguation": False,
                            "flag_reasons": [],
                            "disambiguation_options": [],
                        }
                        st.session_state.detected_items.append(new_item)
                        st.rerun()

        # ── Action buttons ────────────────────────────────────
        col_confirm, col_redetect, col_cancel = st.columns([2, 1, 1])

        with col_confirm:
            if st.button("✅ Confirm All & Get Nutrition",
                         type="primary", use_container_width=True):
                # Filter out any rows where user cleared the name
                valid = [c for c in confirmed if c["food_name"].strip()]
                if not valid:
                    st.warning("Please enter at least one food item name.")
                else:
                    run_nutrition_from_confirmed(valid)
                    st.rerun()

        with col_redetect:
            if st.button("🔁 Re-detect", use_container_width=True,
                         help="Run detection again on the same photo"):
                ib = st.session_state.meal_image_bytes
                im = st.session_state.meal_image_mime
                if ib:
                    st.session_state.meal_stage     = 1
                    st.session_state.detected_items = []
                    run_detection(ib, im)
                    st.rerun()
                else:
                    st.warning("Upload the photo again.")

        with col_cancel:
            if st.button("✖ New Photo", use_container_width=True):
                _reset_meal(); st.rerun()


# ─────────────────────────────────────────────────────────────
# RENDER — STAGE 3: RESULTS
# ─────────────────────────────────────────────────────────────
def render_meal_results():
    items    = st.session_state.meal_items
    totals   = st.session_state.meal_total
    warnings = st.session_state.sanity_warnings

    st.divider()
    st.subheader("📊 Nutrition Analysis")
    for w in warnings: st.warning(f"⚠️ {w}")

    render_summary(totals)
    st.divider()

    st.subheader("🍽️ Confirmed Items")
    for i, item in enumerate(items):
        render_result_card(item, i)

    st.divider()
    render_macros(totals)
    render_micros(totals)

    st.divider()
    col_edit, col_reanalyse, col_new = st.columns(3)

    with col_edit:
        if st.button("✏️ Edit Items", use_container_width=True,
                     help="Go back to review screen to change names or quantities"):
            # Restore detected_items from current meal_items so review is pre-filled
            st.session_state.detected_items = [
                {**item, "portion_size": item["portion_size"],
                          "portion_unit": item["portion_unit"]}
                for item in items
            ]
            st.session_state.meal_stage = 2
            st.rerun()

    with col_reanalyse:
        if st.button("🔁 Re-detect Photo", use_container_width=True,
                     help="Run detection again from scratch"):
            ib = st.session_state.meal_image_bytes
            im = st.session_state.meal_image_mime
            if ib:
                st.session_state.meal_stage     = 1
                st.session_state.detected_items = []
                run_detection(ib, im)
                st.rerun()
            else:
                st.warning("Upload the photo again.")

    with col_new:
        if st.button("🔄 New Photo", use_container_width=True):
            _reset_meal(); st.rerun()

    if st.session_state.debug_mode:
        with st.expander("🐛 Debug: pipeline output"):
            st.json({"meal_items": st.session_state.meal_items,
                     "meal_total": st.session_state.meal_total})


def render_result_card(item: dict, idx: int):
    name       = item.get("food_name", "Unknown")
    grams      = item.get("_grams", "?")
    nutrients  = item.get("nutrients", {})
    source     = item.get("_data_source", "")
    usda_match = item.get("_usda_match", "")
    conf       = item.get("confidence", 0)

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1: st.markdown(f"**{name.title()}**")
        with c2: st.caption(f"~{grams}g")
        with c3:
            kcal = nutrients.get("calories", 0) or 0
            st.caption(f"{kcal:.0f} kcal")

        meta = []
        if source:     meta.append(f"Source: {source}")
        if usda_match: meta.append(f"DB match: '{usda_match}'")
        if meta: st.caption("  •  ".join(meta))

        if item.get("_no_data"):
            st.info("ℹ️ No nutrition data found for this item.")
        if item.get("_error"):
            st.error(f"❌ {item['_error']}")

        if st.session_state.debug_mode:
            with st.expander(f"🐛 Nutrients — {name}"):
                st.json(nutrients)


# ─────────────────────────────────────────────────────────────
# RENDER — LABEL RESULTS  (unchanged)
# ─────────────────────────────────────────────────────────────
def render_label_results():
    data = st.session_state.label_data
    st.divider()
    if "error" in data and not data.get("per_serving"):
        st.error(f"❌ {data['error']}")
        if st.session_state.debug_mode: st.code(data.get("notes", ""))
        return

    quality = data.get("label_quality", "")
    if quality in ("angled", "partial", "low_contrast"):
        st.warning(f"⚠️ Label quality: **{quality}** — some values may be estimated.")

    estimated = data.get("_pass2_estimated", [])
    if estimated:
        st.info(f"ℹ️ Estimated from ingredients (not on label): {', '.join(estimated)}")

    brand   = data.get("brand") or ""
    product = data.get("product_name") or "Unknown Product"
    st.subheader(f"🏷️ {product}" + (f"  —  {brand}" if brand else ""))

    per_serving = data.get("per_serving", {})
    non_null    = {k: v for k, v in per_serving.items() if v is not None}
    if not non_null:
        st.error("Could not extract any values. Try a flatter, better-lit photo.")
        return

    null_fields = [k for k, v in per_serving.items() if v is None]
    if null_fields:
        st.caption(f"Fields not on label: {', '.join(null_fields)}")

    col1, col2 = st.columns([1, 2])
    with col1:
        servings = st.number_input(
            "Servings consumed", min_value=0.1, max_value=10.0,
            value=1.0, step=0.1,
            help="e.g. 0.5 for half, 2.0 for two servings",
        )
    scaled = serving_calculator(per_serving, servings)
    st.session_state.label_total = scaled
    with col2:
        kcal = scaled.get("calories") or 0
        st.metric("Total Calories", f"{kcal:.0f} kcal",
                  delta=f"{servings} serving{'s' if servings!=1 else ''}")

    render_macros(scaled, is_label=True)
    render_micros(scaled, is_label=True)

    if st.button("🔄 Scan Another Label", use_container_width=True):
        st.session_state.label_done = False
        st.session_state.label_data = {}
        st.rerun()

    if st.session_state.debug_mode:
        with st.expander("🐛 Debug: raw label data"): st.json(data)


# ─────────────────────────────────────────────────────────────
# SHARED RENDER HELPERS
# ─────────────────────────────────────────────────────────────
def render_summary(totals: dict):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Total Calories", f"{totals.get('calories',0) or 0:.0f} kcal")
        st.metric("Items", len(st.session_state.meal_items))
    with c2:
        cp = totals.get("_carb_pct", 0) or 0
        pp = totals.get("_protein_pct", 0) or 0
        fp = totals.get("_fat_pct", 0) or 0
        if cp + pp + fp > 0:
            fig = px.pie(
                values=[cp, pp, fp], names=["Carbs", "Protein", "Fat"],
                color_discrete_sequence=["#4A90D9", "#50C878", "#FF6B6B"],
                hole=0.55, title="Macro Split",
            )
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=220,
                              legend=dict(orientation="h", yanchor="bottom", y=-0.35))
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)


def render_macros(data: dict, is_label: bool = False):
    st.subheader("⚡ Macronutrients")
    kcal = data.get("calories", 0) or 1
    c1, c2, c3 = st.columns(3)
    with c1:
        g   = data.get("carbs_g", 0) or 0
        pct = data.get("_carb_pct") or round(g * 4 / kcal * 100, 1)
        st.metric("🌾 Carbohydrates", f"{g:.1f}g")
        st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}% of calories")
        st.caption(f"Fiber: {data.get('fiber_g',0) or 0:.1f}g  •  Sugar: {data.get('sugar_g',0) or 0:.1f}g")
    with c2:
        g   = data.get("protein_g", 0) or 0
        pct = data.get("_protein_pct") or round(g * 4 / kcal * 100, 1)
        st.metric("💪 Protein", f"{g:.1f}g")
        st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}% of calories")
    with c3:
        g   = data.get("fat_g", 0) or 0
        pct = data.get("_fat_pct") or round(g * 9 / kcal * 100, 1)
        st.metric("🥑 Fat", f"{g:.1f}g")
        st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}% of calories")
        st.caption(f"Saturated: {data.get('saturated_fat_g',0) or 0:.1f}g")


def render_micros(data: dict, is_label: bool = False):
    MICROS = {
        "Vitamin A":   ("vit_a_mcg",   "mcg"),
        "Vitamin C":   ("vit_c_mg",    "mg"),
        "Vitamin D":   ("vit_d_mcg",   "mcg"),
        "Vitamin B12": ("vit_b12_mcg", "mcg"),
        "Iron":        ("iron_mg",     "mg"),
        "Calcium":     ("calcium_mg",  "mg"),
        "Zinc":        ("zinc_mg",     "mg"),
        "Magnesium":   ("magnesium_mg","mg"),
        "Potassium":   ("potassium_mg","mg"),
        "Sodium":      ("sodium_mg",   "mg"),
    }
    with st.expander("🧬 Micronutrients — click to expand"):
        c1, c2 = st.columns(2)
        for i, (label, (vk, unit)) in enumerate(MICROS.items()):
            val     = data.get(vk, 0) or 0
            rdi_val = RDI.get(vk, 1) or 1
            rdi_pct = data.get(f"_rdi_{vk}") or round(val / rdi_val * 100, 1)
            rdi_pct = min(float(rdi_pct), 100.0)
            icon    = "🟢" if rdi_pct >= 50 else ("🟡" if rdi_pct >= 20 else "🔴")
            col     = c1 if i % 2 == 0 else c2
            with col:
                st.caption(f"{icon} **{label}**: {val:.2f} {unit}  ({rdi_pct:.0f}% RDI)")
                st.progress(rdi_pct / 100)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🥗 NutriScan AI")
    st.caption("AI-Powered Food Recognition & Nutrition Intelligence")
    st.divider()

    st.subheader("🍽️ Meal Photo Model")
    meal_opts   = ["groq", "gemini", "qwen2"]
    meal_labels = {
        "groq":   "Groq — Llama 4 Scout  (fast, free)",
        "gemini": "Google Gemini 1.5 Flash",
        "qwen2":  "Qwen2-VL 7B  (HuggingFace, slow)",
    }
    sel_meal = st.selectbox(
        "Meal model", options=meal_opts,
        format_func=lambda x: meal_labels[x],
        index=meal_opts.index(st.session_state.meal_provider),
        label_visibility="collapsed",
    )
    if sel_meal != st.session_state.meal_provider:
        st.session_state.meal_provider = sel_meal

    if sel_meal == "qwen2":
        st.warning(
            "⚠️ **Qwen2-VL** takes 10–30 s (cold start may be longer) "
            "and requires HF_TOKEN in .env."
        )
        st.caption("HF_TOKEN: " + ("✅ set" if os.getenv("HF_TOKEN") else "❌ missing"))
    elif sel_meal == "gemini":
        st.caption("GEMINI_API_KEY: " + ("✅ set" if os.getenv("GEMINI_API_KEY") else "❌ missing"))
    else:
        st.caption("GROQ_API_KEY: " + ("✅ set" if os.getenv("GROQ_API_KEY") else "❌ missing"))

    st.divider()
    st.subheader("🏷️ Label OCR Model")
    label_opts   = ["groq", "gemini"]
    label_labels = {
        "groq":   "Groq — Llama 4 Scout  (fast, free)",
        "gemini": "Google Gemini 1.5 Flash",
    }
    sel_label = st.selectbox(
        "Label model", options=label_opts,
        format_func=lambda x: label_labels[x],
        index=label_opts.index(st.session_state.label_provider),
        label_visibility="collapsed",
    )
    if sel_label != st.session_state.label_provider:
        st.session_state.label_provider = sel_label
    st.caption("Qwen2-VL excluded — unreliable for precise number extraction.")

    st.divider()
    st.subheader("🔑 API Status")
    st.write("Groq:",   "✅" if os.getenv("GROQ_API_KEY")   else "❌ missing")
    st.write("USDA:",   "✅" if os.getenv("USDA_API_KEY")   else "⚠️ DEMO_KEY")
    st.write("Gemini:", "✅" if os.getenv("GEMINI_API_KEY") else "⚪ not set")
    st.write("HF:",     "✅" if os.getenv("HF_TOKEN")       else "⚪ not set")

    st.divider()
    st.session_state.debug_mode = st.toggle(
        "🐛 Debug Mode", value=st.session_state.debug_mode,
        help="Shows raw API responses — turn OFF before demo",
    )

    st.divider()
    st.subheader("Coming Soon")
    future = {
        "meal_history":  "📅 Meal History",
        "barcode_scan":  "📷 Barcode Scanner",
        "rdi_dashboard": "📊 Daily RDI Dashboard",
        "pdf_export":    "📄 Export PDF",
        "calorie_goals": "🎯 Calorie Goals",
        "diet_tagging":  "🏷️ Diet Tags",
    }
    for flag, label in future.items():
        if FEATURE_FLAGS.get(flag):
            st.write(label)
        else:
            st.caption(f"~~{label}~~")


# ─────────────────────────────────────────────────────────────
# MAIN UI  (all functions defined above)
# ─────────────────────────────────────────────────────────────
st.title("🥗 NutriScan AI")
st.caption("Upload a meal photo or scan a food label for instant nutrition analysis.")

tab_meal, tab_label = st.tabs(["🍽️ Analyze Meal Photo", "🏷️ Scan Food Label"])

# ── Meal tab — 3-stage flow ───────────────────────────────────
with tab_meal:
    stage = st.session_state.meal_stage

    # Stage 1: Upload
    if stage == 1:
        cu, cp = st.columns([1, 1])
        with cu:
            st.subheader("Upload Your Meal")
            src = st.radio("src", ["Upload file", "Use camera"],
                           horizontal=True, label_visibility="collapsed")
            img_file = None
            if src == "Upload file":
                img_file = st.file_uploader(
                    "Choose meal photo",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="meal_upload",
                )
            else:
                img_file = st.camera_input("Take a photo", key="meal_camera")
        with cp:
            if img_file:
                st.image(img_file, caption="Your meal", use_container_width=True)

        if img_file:
            if st.button("🔍 Analyze Meal", type="primary", use_container_width=True):
                ib = _read(img_file)
                mi = _mime(img_file)
                st.session_state.meal_image_bytes = ib
                st.session_state.meal_image_mime  = mi
                run_detection(ib, mi)
                st.rerun()

    # Stage 2: Review & Confirm
    elif stage == 2:
        render_review_screen()

    # Stage 3: Results
    elif stage == 3:
        render_meal_results()


# ── Label tab ─────────────────────────────────────────────────
with tab_label:
    if not st.session_state.label_done:
        cu2, cp2 = st.columns([1, 1])
        with cu2:
            st.subheader("Upload Food Label")
            st.caption("Photograph the nutrition facts panel. Hindi/regional labels supported.")
            src2 = st.radio("src2", ["Upload file", "Use camera"],
                            horizontal=True, label_visibility="collapsed",
                            key="label_src")
            lbl_file = None
            if src2 == "Upload file":
                lbl_file = st.file_uploader(
                    "Choose label photo",
                    type=["jpg", "jpeg", "png", "webp"],
                    key="label_upload",
                )
            else:
                lbl_file = st.camera_input("Take a photo", key="label_camera")
        with cp2:
            if lbl_file:
                st.image(lbl_file, caption="Food label", use_container_width=True)

        if lbl_file:
            if st.button("📖 Extract Label Info", type="primary", use_container_width=True):
                run_label_extraction(_read(lbl_file), _mime(lbl_file))
                st.rerun()

    if st.session_state.label_done and st.session_state.label_data:
        render_label_results()
