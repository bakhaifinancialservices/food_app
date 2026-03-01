# app.py
# ─────────────────────────────────────────────────────────────
# AI-Powered Food Recognition & Nutrition Intelligence Platform
# Run locally: streamlit run app.py
# ─────────────────────────────────────────────────────────────

import logging
import os

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

from utils.vision import detect_foods
from utils.portions import to_grams
from utils.normalize import normalize_name
from utils.nutrition import lookup_nutrition, scale_nutrients, aggregate_meal, RDI
from utils.validate import run_validation, nutrition_sanity_check
from utils.label_ocr import extract_label, serving_calculator

# ── Feature flags ─────────────────────────────────────────────
FEATURE_FLAGS = {
    "meal_history":   False,
    "barcode_scan":   False,
    "rdi_dashboard":  False,
    "pdf_export":     False,
    "calorie_goals":  False,
    "api_endpoint":   False,
    "multi_language": False,
    "diet_tagging":   False,
}

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NutriScan AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────
def init_session():
    defaults = {
        "meal_items":       [],
        "meal_total":       {},
        "sanity_warnings":  [],
        "corrections":      {},   # {original_name: corrected_name}
        "label_data":       {},
        "label_total":      {},
        "analysis_done":    False,
        "label_done":       False,
        "debug_mode":       False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _get_mime(file) -> str:
    name = getattr(file, "name", "image.jpg").lower()
    if name.endswith(".png"):  return "image/png"
    if name.endswith(".webp"): return "image/webp"
    return "image/jpeg"

def _read_file(file) -> bytes:
    """Read uploaded file safely — seeks to 0 first in case already read."""
    file.seek(0)
    return file.read()

# ─────────────────────────────────────────────────────────────
# PIPELINE FUNCTIONS  — defined BEFORE the UI blocks that call them
# ─────────────────────────────────────────────────────────────

def _run_meal_analysis(image_file):
    """Full meal analysis: detect → apply corrections → validate → nutrition → aggregate."""
    st.session_state.analysis_done  = False
    st.session_state.meal_items     = []
    st.session_state.meal_total     = {}
    st.session_state.sanity_warnings = []

    image_bytes = _read_file(image_file)   # safe read with seek(0)

    # Step 1: Detect foods via Groq Llama 4 Scout
    with st.spinner("🔍 Detecting food items..."):
        try:
            detected = detect_foods(image_bytes, _get_mime(image_file))
            logger.info(f"[app] Detected {len(detected)} item(s)")
        except Exception as e:
            st.error(f"❌ Food detection failed: {e}")
            logger.error("[app] Detection error", exc_info=True)
            return

    if not detected:
        st.warning("⚠️ No food items detected. Try a clearer photo with better lighting.")
        return

    # Step 2: Apply any saved user corrections from previous run
    corrections = st.session_state.corrections
    if corrections:
        for item in detected:
            original = item["food_name"]
            if original in corrections:
                item["food_name"]      = corrections[original]
                item["_was_corrected"] = True
                item["_original_name"] = original
                logger.info(f"[app] Correction applied: '{original}' → '{item['food_name']}'")

    # Step 3: Validate and flag hallucinations
    with st.spinner("🧠 Checking detections..."):
        try:
            enriched, _ = run_validation(detected)
        except Exception as e:
            st.warning(f"⚠️ Validation issue: {e} — continuing anyway.")
            enriched = detected
            logger.error("[app] Validation error", exc_info=True)

    # Step 4: Nutrition lookup for every item
    with st.spinner("📊 Fetching nutrition data..."):
        for item in enriched:
            try:
                grams      = to_grams(item["food_name"], item["portion_size"], item["portion_unit"])
                normalized = normalize_name(item["food_name"])
                raw        = lookup_nutrition(normalized)

                item["_grams"]           = grams
                item["_normalized_name"] = normalized

                if raw:
                    item["nutrients"]    = scale_nutrients(raw, grams)
                    item["_data_source"] = raw.get("_source", "unknown")
                    item["_usda_match"]  = raw.get("_usda_description", "")
                else:
                    item["nutrients"] = {}
                    item["_no_data"]  = True
                    logger.warning(f"[app] No nutrition data for '{item['food_name']}'")

            except Exception as e:
                item["nutrients"] = {}
                item["_error"]    = str(e)
                logger.error(f"[app] Nutrition error for '{item['food_name']}'", exc_info=True)

    # Step 5: Aggregate meal totals
    try:
        meal_total = aggregate_meal(enriched)
        warnings   = nutrition_sanity_check(enriched, meal_total)

        st.session_state.meal_items      = enriched
        st.session_state.meal_total      = meal_total
        st.session_state.sanity_warnings = warnings
        st.session_state.analysis_done   = True
        logger.info(f"[app] Done — {meal_total.get('calories', 0):.0f} kcal total")
    except Exception as e:
        st.error(f"❌ Aggregation failed: {e}")
        logger.error("[app] Aggregation error", exc_info=True)


def _run_label_extraction(label_file):
    """Label OCR: extract structured nutrition facts from a label photo."""
    st.session_state.label_done  = False
    st.session_state.label_data  = {}
    st.session_state.label_total = {}

    image_bytes = _read_file(label_file)   # safe read

    with st.spinner("📖 Reading nutrition label..."):
        try:
            label_data = extract_label(image_bytes, _get_mime(label_file))
            st.session_state.label_data = label_data
            st.session_state.label_done = True
            logger.info(f"[app] Label done — quality={label_data.get('label_quality')}")
        except Exception as e:
            st.error(f"❌ Label extraction failed: {e}")
            logger.error("[app] Label error", exc_info=True)


# ─────────────────────────────────────────────────────────────
# RENDER FUNCTIONS  — also defined before the UI blocks
# ─────────────────────────────────────────────────────────────

def render_meal_results():
    items    = st.session_state.meal_items
    totals   = st.session_state.meal_total
    warnings = st.session_state.sanity_warnings

    st.divider()
    st.subheader("📊 Analysis Results")

    for w in warnings:
        st.warning(f"⚠️ {w}")

    render_meal_summary(totals)
    st.divider()

    st.subheader("🍽️ Detected Items")
    for i, item in enumerate(items):
        render_item_card(item, i)

    # Reset — clears results so user can upload a new photo
    if st.button("🔄 Analyze Another Photo", use_container_width=True):
        st.session_state.analysis_done  = False
        st.session_state.meal_items     = []
        st.session_state.meal_total     = {}
        st.session_state.corrections    = {}
        st.session_state.sanity_warnings = []
        st.rerun()

    st.divider()
    render_macro_breakdown(totals)
    render_micro_breakdown(totals)

    if st.session_state.debug_mode:
        with st.expander("🐛 Debug: raw pipeline output"):
            st.json({
                "meal_items":  st.session_state.meal_items,
                "meal_total":  st.session_state.meal_total,
                "corrections": st.session_state.corrections,
            })


def render_label_results():
    label_data = st.session_state.label_data

    st.divider()

    if "error" in label_data and not label_data.get("per_serving"):
        st.error(f"❌ {label_data['error']}")
        if st.session_state.debug_mode:
            st.code(label_data.get("notes", ""), language="text")
        return

    quality = label_data.get("label_quality", "unknown")
    if quality in ("angled", "partial", "low_contrast"):
        st.warning(f"⚠️ Label quality: **{quality}**. Some values may be missing.")

    product = label_data.get("product_name") or "Unknown Product"
    st.subheader(f"🏷️ {product}")

    per_serving = label_data.get("per_serving", {})
    non_null    = {k: v for k, v in per_serving.items() if v is not None}
    if not non_null:
        st.error("Could not extract any nutrition values. Try a clearer, flatter photo.")
        return

    null_fields = [k for k, v in per_serving.items() if v is None]
    if null_fields:
        st.info(f"ℹ️ Fields not visible on label: {', '.join(null_fields)}")

    col1, col2 = st.columns([1, 2])
    with col1:
        servings = st.number_input(
            "Servings consumed",
            min_value=0.5, max_value=10.0, value=1.0, step=0.5,
            help="Adjust to match how much you actually ate",
        )

    scaled = serving_calculator(per_serving, servings)
    st.session_state.label_total = scaled

    with col2:
        kcal = scaled.get("calories") or 0
        st.metric("Total Calories", f"{kcal:.0f} kcal",
                  delta=f"{servings} serving{'s' if servings != 1 else ''}")

    render_macro_breakdown(scaled, is_label=True)
    render_micro_breakdown(scaled, is_label=True)

    if st.button("🔄 Scan Another Label", use_container_width=True):
        st.session_state.label_done  = False
        st.session_state.label_data  = {}
        st.session_state.label_total = {}
        st.rerun()

    if st.session_state.debug_mode:
        with st.expander("🐛 Debug: raw label data"):
            st.json(label_data)


def render_meal_summary(totals: dict):
    col1, col2 = st.columns([1, 1])

    with col1:
        kcal  = totals.get("calories", 0) or 0
        items = len(st.session_state.meal_items)
        st.metric("Total Calories", f"{kcal:.0f} kcal")
        st.metric("Items Detected", items)

    with col2:
        c = totals.get("_carb_pct", 0) or 0
        p = totals.get("_protein_pct", 0) or 0
        f = totals.get("_fat_pct", 0) or 0

        if c + p + f > 0:
            fig = px.pie(
                values=[c, p, f],
                names=["Carbs", "Protein", "Fat"],
                color_discrete_sequence=["#4A90D9", "#50C878", "#FF6B6B"],
                hole=0.55,
                title="Macro Split (% of calories)",
            )
            fig.update_layout(
                margin=dict(t=40, b=0, l=0, r=0),
                height=220,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)


def render_item_card(item: dict, index: int):
    name         = item.get("food_name", "Unknown")
    conf         = item.get("confidence", 0)
    grams        = item.get("_grams", "?")
    is_flagged   = item.get("is_flagged", False)
    needs_dis    = item.get("needs_disambiguation", False)
    flag_reasons = item.get("flag_reasons", [])
    nutrients    = item.get("nutrients", {})
    source       = item.get("_data_source", "")
    usda_match   = item.get("_usda_match", "")
    corrected    = item.get("_was_corrected", False)
    original     = item.get("_original_name", "")

    with st.container(border=True):
        # Header
        if conf >= 0.80:   badge = f"🟢 {conf:.0%}"
        elif conf >= 0.65: badge = f"🟡 {conf:.0%}"
        else:              badge = f"🔴 {conf:.0%}"

        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            label = f"**{name.title()}**"
            if corrected:
                label += f"  *(was '{original}')*"
            st.markdown(label)
        with c2:
            st.markdown(badge)
        with c3:
            st.caption(f"~{grams}g")

        # Source info
        parts = []
        if source:     parts.append(f"Source: {source}")
        if usda_match: parts.append(f"Matched: '{usda_match}'")
        if parts:
            st.caption("  •  ".join(parts))

        # Warnings
        for reason in flag_reasons:
            st.warning(f"⚠️ {reason}")
        if item.get("_no_data"):
            st.info("ℹ️ No nutrition data found. Correct the name below and click Analyze again.")
        if item.get("_error"):
            st.error(f"❌ Nutrition error: {item['_error']}")

        # Disambiguation radio — user picks the correct food
        if needs_dis or is_flagged:
            options = item.get("disambiguation_options", [name])
            if options:
                key = f"disambig_{index}_{name}"
                saved = st.session_state.corrections.get(name, name)
                default_idx = options.index(saved) if saved in options else 0

                choice = st.radio(
                    "✏️ Confirm or correct:",
                    options,
                    index=default_idx,
                    key=key,
                    horizontal=True,
                )
                if choice != name:
                    st.session_state.corrections[name] = choice
                    st.info(f"Saved correction: '{name}' → '{choice}'. "
                            "Click **Analyze Meal** again to update nutrition.")
                elif name in st.session_state.corrections:
                    del st.session_state.corrections[name]   # user reverted

        # Per-item calorie contribution
        kcal = nutrients.get("calories", 0) or 0
        if kcal:
            st.caption(f"Contributes {kcal:.0f} kcal to meal total")

        if st.session_state.debug_mode:
            with st.expander(f"🐛 Raw nutrients — {name}"):
                st.json(nutrients)


def render_macro_breakdown(data: dict, is_label: bool = False):
    st.subheader("⚡ Macronutrients")
    kcal = data.get("calories", 0) or 1  # avoid divide-by-zero

    c1, c2, c3 = st.columns(3)

    with c1:
        g    = data.get("carbs_g", 0) or 0
        pct  = data.get("_carb_pct") or round(g * 4 / kcal * 100, 1)
        fiber = data.get("fiber_g", 0) or 0
        sugar = data.get("sugar_g", 0) or 0
        st.metric("🌾 Carbohydrates", f"{g:.1f}g")
        st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}% of calories")
        st.caption(f"Fiber: {fiber:.1f}g  •  Sugar: {sugar:.1f}g")

    with c2:
        g   = data.get("protein_g", 0) or 0
        pct = data.get("_protein_pct") or round(g * 4 / kcal * 100, 1)
        st.metric("💪 Protein", f"{g:.1f}g")
        st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}% of calories")

    with c3:
        g   = data.get("fat_g", 0) or 0
        pct = data.get("_fat_pct") or round(g * 9 / kcal * 100, 1)
        sat = data.get("saturated_fat_g", 0) or 0
        st.metric("🥑 Fat", f"{g:.1f}g")
        st.progress(min(pct / 100, 1.0), text=f"{pct:.1f}% of calories")
        st.caption(f"Saturated: {sat:.1f}g")


def render_micro_breakdown(data: dict, is_label: bool = False):
    MICROS = {
        "Vitamin A":   ("vit_a_mcg",    "mcg"),
        "Vitamin C":   ("vit_c_mg",     "mg"),
        "Vitamin D":   ("vit_d_mcg",    "mcg"),
        "Vitamin B12": ("vit_b12_mcg",  "mcg"),
        "Iron":        ("iron_mg",      "mg"),
        "Calcium":     ("calcium_mg",   "mg"),
        "Zinc":        ("zinc_mg",      "mg"),
        "Magnesium":   ("magnesium_mg", "mg"),
        "Potassium":   ("potassium_mg", "mg"),
        "Sodium":      ("sodium_mg",    "mg"),
    }

    with st.expander("🧬 Micronutrients — click to expand"):
        c1, c2 = st.columns(2)
        for i, (label, (val_key, unit)) in enumerate(MICROS.items()):
            val     = data.get(val_key, 0) or 0
            rdi_val = RDI.get(val_key, 1) or 1
            rdi_key = f"_rdi_{val_key}"
            rdi_pct = data.get(rdi_key) or round((val / rdi_val) * 100, 1)
            rdi_pct = min(float(rdi_pct), 100.0)

            if rdi_pct >= 50:   icon = "🟢"
            elif rdi_pct >= 20: icon = "🟡"
            else:               icon = "🔴"

            col = c1 if i % 2 == 0 else c2
            with col:
                st.caption(f"{icon} **{label}**: {val:.2f} {unit}  ({rdi_pct:.0f}% RDI)")
                st.progress(rdi_pct / 100)


# ─────────────────────────────────────────────────────────────
# SIDEBAR  — rendered after all functions are defined
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🥗 NutriScan AI")
    st.caption("AI-Powered Food Recognition & Nutrition Intelligence")
    st.divider()

    st.session_state.debug_mode = st.toggle(
        "🐛 Debug Mode",
        value=st.session_state.debug_mode,
        help="Shows raw API responses — turn OFF before demo",
    )
    st.divider()

    st.subheader("API Status")
    groq_key = os.getenv("GROQ_API_KEY", "")
    usda_key = os.getenv("USDA_API_KEY", "")
    st.write("Groq API:", "✅ Set" if groq_key else "❌ Missing — add to .env")
    st.write("USDA API:", "✅ Set" if usda_key else "⚠️ Will use DEMO_KEY (limited)")
    if not groq_key:
        st.error("GROQ_API_KEY is required. Add it to .env and restart.")

    st.divider()
    st.subheader("Coming Soon")
    future = {
        "meal_history":  "📅 Meal History",
        "barcode_scan":  "📷 Barcode Scanner",
        "rdi_dashboard": "📊 Daily RDI Dashboard",
        "pdf_export":    "📄 Export as PDF",
        "calorie_goals": "🎯 Calorie & Macro Goals",
        "diet_tagging":  "🏷️ Diet Tags (Keto, Diabetic…)",
    }
    for flag, label in future.items():
        if FEATURE_FLAGS.get(flag):
            st.write(label)
        else:
            st.caption(f"~~{label}~~")

# ─────────────────────────────────────────────────────────────
# MAIN UI  — all functions already defined above, safe to call
# ─────────────────────────────────────────────────────────────
st.title("🥗 NutriScan AI")
st.caption("Upload a meal photo or scan a food label for instant macro & micronutrient analysis.")

tab_meal, tab_label = st.tabs(["🍽️ Analyze Meal Photo", "🏷️ Scan Food Label"])

# ── Tab 1: Meal photo ─────────────────────────────────────────
with tab_meal:
    if not st.session_state.analysis_done:
        col_up, col_prev = st.columns([1, 1])
        with col_up:
            st.subheader("Upload Your Meal")
            src = st.radio("Source", ["Upload file", "Use camera"],
                           horizontal=True, label_visibility="collapsed")
            image_file = None
            if src == "Upload file":
                image_file = st.file_uploader(
                    "Choose meal photo", type=["jpg", "jpeg", "png", "webp"],
                    key="meal_upload")
            else:
                image_file = st.camera_input("Take a photo", key="meal_camera")

        with col_prev:
            if image_file:
                st.image(image_file, caption="Your meal", use_container_width=True)

        if image_file:
            if st.button("🔍 Analyze Meal", type="primary", use_container_width=True):
                _run_meal_analysis(image_file)   # safe — defined above

    if st.session_state.analysis_done and st.session_state.meal_items:
        render_meal_results()                    # safe — defined above

# ── Tab 2: Food label ─────────────────────────────────────────
with tab_label:
    if not st.session_state.label_done:
        col_up2, col_prev2 = st.columns([1, 1])
        with col_up2:
            st.subheader("Upload Food Label")
            st.caption("Photograph the nutrition facts panel of any packaged food.")
            src2 = st.radio("Source", ["Upload file", "Use camera"],
                            horizontal=True, label_visibility="collapsed",
                            key="label_source")
            label_file = None
            if src2 == "Upload file":
                label_file = st.file_uploader(
                    "Choose label photo", type=["jpg", "jpeg", "png", "webp"],
                    key="label_upload")
            else:
                label_file = st.camera_input("Take a photo", key="label_camera")

        with col_prev2:
            if label_file:
                st.image(label_file, caption="Food label", use_container_width=True)

        if label_file:
            if st.button("📖 Extract Label Info", type="primary", use_container_width=True):
                _run_label_extraction(label_file)  # safe — defined above

    if st.session_state.label_done and st.session_state.label_data:
        render_label_results()                     # safe — defined above
