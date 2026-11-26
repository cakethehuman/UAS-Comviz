"""Streamlit Head-to-Head Model Arena for EfficientNetB0 vs Vision Transformer (ViT)."""

import base64
import hashlib
import html
from io import BytesIO
from typing import Any

import pandas as pd
from PIL import Image, ImageOps
import requests
import streamlit as st

DEFAULT_API_BASE = "http://127.0.0.1:8000"
CLASS_LABELS = [
    "Bean",
    "Bitter melon",
    "Cucumber",
    "Green Chili",
    "Ladies finger",
    "Pointed gourd",
]

st.set_page_config(
    page_title="EfficientNetB0 vs ViT · Model Arena",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: #0B0F19;
    color: #F1F5F9;
}
[data-testid="stHeader"] {
    background-color: rgba(11, 15, 25, 0.85);
}
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid #1E293B;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #111827;
    border: 1px dashed #334155;
    border-radius: 12px;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container {
    max-width: 1320px;
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}

/* Header */
.arena-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    padding-bottom: 1rem;
    margin-bottom: 1.25rem;
    border-bottom: 1px solid #1E293B;
    flex-wrap: wrap;
}
.arena-title {
    margin: 0;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #F8FAFC;
    line-height: 1.2;
}
.arena-subtitle {
    margin: 0.3rem 0 0 0;
    font-size: 0.92rem;
    color: #94A3B8;
}
.status-cluster {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.32rem 0.72rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid transparent;
}
.pill-ready {
    background: rgba(16, 185, 129, 0.12);
    color: #6EE7B7;
    border-color: rgba(16, 185, 129, 0.30);
}
.pill-waiting {
    background: rgba(99, 102, 241, 0.14);
    color: #C7D2FE;
    border-color: rgba(99, 102, 241, 0.32);
}
.pill-offline {
    background: rgba(239, 68, 68, 0.14);
    color: #FCA5A5;
    border-color: rgba(239, 68, 68, 0.35);
}
.pill-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.dot-emerald { background: #10B981; }
.dot-indigo { background: #6366F1; }
.dot-red { background: #EF4444; }

/* Cards */
.surface-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}
.preview-frame {
    background: #0B1220;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 0.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 180px;
}
.preview-frame img {
    max-height: 255px;
    max-width: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
    border-radius: 6px;
}
.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.75rem;
}
.meta-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #CBD5E1;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.2rem 0.52rem;
}
.class-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.65rem;
}
.class-chip {
    font-size: 0.76rem;
    font-weight: 500;
    color: #94A3B8;
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 9999px;
    padding: 0.22rem 0.62rem;
}
.class-chip-ef {
    background: rgba(16, 185, 129, 0.15);
    color: #6EE7B7;
    border-color: rgba(16, 185, 129, 0.45);
    font-weight: 600;
}
.class-chip-vit {
    background: rgba(99, 102, 241, 0.18);
    color: #C7D2FE;
    border-color: rgba(99, 102, 241, 0.45);
    font-weight: 600;
}
.class-chip-both {
    background: #F8FAFC;
    color: #0B0F19;
    border-color: #F8FAFC;
    font-weight: 600;
}

/* Banners */
.verdict-banner {
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.9rem 0 1rem 0;
    border: 1px solid #1F2937;
    background: #111827;
    color: #F1F5F9;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}
.verdict-agree { border-left: 4px solid #10B981; background: rgba(16, 185, 129, 0.10); border-color: rgba(16, 185, 129, 0.30); }
.verdict-disagree { border-left: 4px solid #F59E0B; background: rgba(245, 158, 11, 0.10); border-color: rgba(245, 158, 11, 0.30); }
.verdict-single { border-left: 4px solid #818CF8; background: rgba(99, 102, 241, 0.10); border-color: rgba(99, 102, 241, 0.28); }
.caution-banner {
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 1rem 0;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.32);
    border-left: 4px solid #F59E0B;
    color: #FDE68A;
    font-size: 0.86rem;
}

/* Arena Model Cards */
.model-card {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 14px;
    padding: 1.25rem 1.3rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    margin-bottom: 0.65rem;
}
.model-card-ef { border-top: 4px solid #10B981; }
.model-card-vit { border-top: 4px solid #6366F1; }

.model-card-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 0.9rem;
}
.model-name {
    font-size: 1.12rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0;
}
.model-arch {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 0.18rem;
}
.badge-ef {
    background: rgba(16, 185, 129, 0.14);
    color: #6EE7B7;
    border: 1px solid rgba(16, 185, 129, 0.35);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.22rem 0.6rem;
    border-radius: 9999px;
    white-space: nowrap;
}
.badge-vit {
    background: rgba(99, 102, 241, 0.16);
    color: #C7D2FE;
    border: 1px solid rgba(99, 102, 241, 0.35);
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.22rem 0.6rem;
    border-radius: 9999px;
    white-space: nowrap;
}

.hero-prediction {
    background: #0B1220;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.95rem;
}
.hero-kicker {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
}
.hero-label {
    font-size: 1.55rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0.15rem 0 0.25rem 0;
    line-height: 1.15;
}
.hero-sub {
    font-size: 0.82rem;
    color: #CBD5E1;
    font-family: 'JetBrains Mono', monospace;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.55rem;
    margin-bottom: 1rem;
}
.metric-tile {
    background: #0B1220;
    border: 1px solid #1E293B;
    border-radius: 9px;
    padding: 0.55rem 0.7rem;
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #94A3B8;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.98rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-top: 0.12rem;
}

/* Probability bars */
.prob-section-title {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.55rem;
}
.prob-row {
    margin-bottom: 0.52rem;
}
.prob-row-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.83rem;
    margin-bottom: 0.2rem;
}
.prob-label {
    color: #CBD5E1;
    font-weight: 500;
}
.prob-label-top {
    font-weight: 700;
    color: #F8FAFC;
}
.prob-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.79rem;
    color: #CBD5E1;
}
.prob-track {
    width: 100%;
    height: 8px;
    background: #1E293B;
    border-radius: 9999px;
    overflow: hidden;
}
.prob-fill-ef {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #10B981, #059669);
}
.prob-fill-vit {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, #818CF8, #4F46E5);
}
.prob-fill-muted {
    height: 100%;
    border-radius: 9999px;
    background: #475569;
}

/* Standby / awaiting card */
.standby-box {
    background: rgba(99, 102, 241, 0.08);
    border: 1px dashed rgba(129, 140, 248, 0.40);
    border-radius: 11px;
    padding: 1.15rem 1.05rem;
    margin-bottom: 0.9rem;
}
.standby-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: #C7D2FE;
    margin: 0 0 0.35rem 0;
}
.standby-text {
    font-size: 0.84rem;
    color: #CBD5E1;
    line-height: 1.5;
    margin: 0;
}
.code-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    background: #1E293B;
    color: #A5B4FC;
    padding: 0.14rem 0.42rem;
    border-radius: 5px;
    border: 1px solid #334155;
}
.arch-list {
    margin: 0.7rem 0 0 0;
    padding-left: 1.1rem;
    font-size: 0.82rem;
    color: #CBD5E1;
    line-height: 1.55;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _extract_error_message(response: requests.Response) -> str:
    """Extract clean human-readable error messages from FastAPI JSON errors."""
    try:
        payload = response.json()
    except ValueError:
        return f"Server returned HTTP {response.status_code}: {response.text[:240]}"

    if isinstance(payload, dict):
        if isinstance(payload.get("error"), dict):
            code = payload["error"].get("code", "ERROR")
            msg = payload["error"].get("message", "Request failed.")
            return f"[{code}] {msg}"
        detail = payload.get("detail")
        if isinstance(detail, dict):
            code = detail.get("code", "ERROR")
            msg = detail.get("message", str(detail))
            return f"[{code}] {msg}"
        if isinstance(detail, str):
            return detail
    return f"Request failed with HTTP {response.status_code}."


def fetch_health(api_base: str) -> tuple[dict[str, Any] | None, str | None]:
    """Check backend health and loaded model readiness."""
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/health", timeout=4)
        if resp.ok:
            return resp.json(), None
        return None, _extract_error_message(resp)
    except requests.ConnectionError:
        return None, "Backend server is unreachable. Start FastAPI on port 8000."
    except requests.Timeout:
        return None, "Backend health check timed out."
    except requests.RequestException as exc:
        return None, f"Backend request failed: {exc}"


def fetch_models(api_base: str) -> tuple[dict[str, Any] | None, str | None]:
    """Fetch model registry metadata."""
    try:
        resp = requests.get(f"{api_base.rstrip('/')}/models", timeout=4)
        if resp.ok:
            return resp.json(), None
        return None, _extract_error_message(resp)
    except requests.RequestException as exc:
        return None, str(exc)


def reload_models(api_base: str) -> tuple[dict[str, Any] | None, str | None]:
    """Trigger backend weight discovery & hot-reload."""
    try:
        resp = requests.post(f"{api_base.rstrip('/')}/models/reload", timeout=60)
        if resp.ok:
            return resp.json(), None
        return None, _extract_error_message(resp)
    except requests.RequestException as exc:
        return None, f"Reload failed: {exc}"


def run_comparison(
    api_base: str,
    filename: str,
    file_bytes: bytes,
    content_type: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Send image once to POST /predict/compare for side-by-side inference."""
    try:
        files = {
            "file": (
                filename or "upload.jpg",
                file_bytes,
                content_type or "application/octet-stream",
            )
        }
        resp = requests.post(
            f"{api_base.rstrip('/')}/predict/compare",
            files=files,
            timeout=45,
        )
        if resp.ok:
            return resp.json(), None
        return None, _extract_error_message(resp)
    except requests.ConnectionError:
        return None, "Could not reach backend API. Verify FastAPI is running."
    except requests.Timeout:
        return None, "Prediction request timed out after 45s."
    except requests.RequestException as exc:
        return None, f"Prediction request failed: {exc}"


def _build_preview_data_uri(file_bytes: bytes) -> tuple[str | None, dict[str, Any] | None]:
    """Safely render EXIF-corrected preview URI and local image specs."""
    try:
        with Image.open(BytesIO(file_bytes)) as img:
            img_format = (img.format or "UNKNOWN").upper()
            transposed = ImageOps.exif_transpose(img)
            width, height = transposed.size
            if "A" in transposed.mode or "transparency" in transposed.info:
                rgba = transposed.convert("RGBA")
                bg = Image.new("RGB", rgba.size, (11, 18, 32))
                bg.paste(rgba, mask=rgba.split()[3])
                preview_rgb = bg
            else:
                preview_rgb = transposed.convert("RGB")
            preview_rgb.thumbnail((900, 900))
            buf = BytesIO()
            preview_rgb.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}", {
                "format": img_format,
                "width": width,
                "height": height,
            }
    except Exception:
        return None, None


def _render_probability_bars(
    probabilities: list[dict[str, Any]],
    accent_class: str,
) -> str:
    """Build HTML-escaped sorted probability bars with ±std dispersion."""
    rows: list[str] = []
    for rank, item in enumerate(probabilities):
        label = html.escape(str(item["label"]))
        prob = float(item["probability"])
        pct = prob * 100.0
        std_val = item.get("std")
        std_str = f" ± {float(std_val) * 100.0:.1f}%" if std_val is not None else ""
        bar_width = max(1.5, min(100.0, pct)) if pct > 0.0 else 0.0
        label_cls = "prob-label prob-label-top" if rank == 0 else "prob-label"
        fill_cls = accent_class if rank == 0 else "prob-fill-muted"

        rows.append(
            f'<div class="prob-row">'
            f'  <div class="prob-row-top">'
            f'    <span class="{label_cls}">{label}</span>'
            f'    <span class="prob-num">{pct:.1f}%{html.escape(std_str)}</span>'
            f'  </div>'
            f'  <div class="prob-track">'
            f'    <div class="{fill_cls}" style="width: {bar_width:.1f}%;"></div>'
            f'  </div>'
            f'</div>'
        )
    return "".join(rows)


def _render_ready_model_card(
    result: dict[str, Any],
    side_role: str,
    card_accent_cls: str,
    badge_cls: str,
    bar_fill_cls: str,
) -> None:
    """Render an active model column (Left EfficientNetB0 or Right ViT)."""
    display_name = html.escape(str(result.get("display_name", "")))
    architecture = html.escape(str(result.get("architecture", "")))
    pred_label = html.escape(str(result.get("predicted_label", "—")))
    conf_pct = float(result.get("confidence") or 0.0) * 100.0
    margin_pct = float(result.get("top2_margin") or 0.0) * 100.0
    entropy_val = float(result.get("normalized_entropy") or 0.0)
    latency_ms = float(result.get("inference_time_ms") or 0.0)
    fold_count = int(result.get("fold_count") or 0)
    fold_agreement = html.escape(str(result.get("fold_agreement") or "—"))

    ensemble_tag = f"{fold_count}-Fold Ensemble" if fold_count > 1 else "Single Checkpoint"
    bars_html = _render_probability_bars(result.get("probabilities", []), bar_fill_cls)

    card_html = (
        f'<div class="model-card {card_accent_cls}">'
        f'  <div class="model-card-head">'
        f'    <div>'
        f'      <div class="model-name">{display_name}</div>'
        f'      <div class="model-arch">{architecture}</div>'
        f'    </div>'
        f'    <span class="{badge_cls}">{html.escape(side_role)} · {html.escape(ensemble_tag)}</span>'
        f'  </div>'
        f'  <div class="hero-prediction">'
        f'    <div class="hero-kicker">Top-1 Prediction</div>'
        f'    <div class="hero-label">{pred_label}</div>'
        f'    <div class="hero-sub">Latency {latency_ms:.1f} ms · Fold consensus {fold_agreement}</div>'
        f'  </div>'
        f'  <div class="metric-strip">'
        f'    <div class="metric-tile">'
        f'      <div class="metric-label">Confidence</div>'
        f'      <div class="metric-value">{conf_pct:.1f}%</div>'
        f'    </div>'
        f'    <div class="metric-tile">'
        f'      <div class="metric-label">Top-2 Margin</div>'
        f'      <div class="metric-value">+{margin_pct:.1f}%</div>'
        f'    </div>'
        f'    <div class="metric-tile">'
        f'      <div class="metric-label">Entropy</div>'
        f'      <div class="metric-value">{entropy_val:.2f}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="prob-section-title">Class Probability Distribution (Mean ± Ensemble Std)</div>'
        f'  {bars_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    fold_preds = result.get("fold_predictions") or []
    if fold_preds:
        with st.expander(f"{result.get('display_name')} · {fold_count}-Fold Ensemble Diagnostics", expanded=False):
            df_folds = pd.DataFrame(
                [
                    {
                        "Fold": f"Fold {fp['fold']}",
                        "Weight File": fp["weight_file"],
                        "Prediction": fp["predicted_label"],
                        "Confidence": f"{float(fp['confidence']) * 100.0:.2f}%",
                    }
                    for fp in fold_preds
                ]
            )
            st.dataframe(df_folds, width="stretch", hide_index=True)


def _render_standby_model_card(
    model_key: str,
    model_info: dict[str, Any] | None,
    result: dict[str, Any] | None,
    side_role: str,
    card_accent_cls: str,
    badge_cls: str,
    api_base: str,
) -> None:
    """Render an honest awaiting-weights card when ViT (or another slot) has no weights yet."""
    is_vit = model_key == "vit"
    display_name = html.escape(
        str(
            (result or {}).get("display_name")
            or (model_info or {}).get("display_name")
            or ("Vision Transformer (ViT)" if is_vit else "EfficientNetB0")
        )
    )
    architecture = html.escape(
        str(
            (result or {}).get("architecture")
            or (model_info or {}).get("architecture")
            or ("Transformer Self-Attention over Image Patches" if is_vit else "Convolutional Neural Network")
        )
    )
    reason = html.escape(
        str(
            (result or {}).get("reason")
            or (model_info or {}).get("reason")
            or "No checkpoint weights found in backend/model/."
        )
    )

    st.markdown(
        f'<div class="model-card {card_accent_cls}">'
        f'  <div class="model-card-head">'
        f'    <div>'
        f'      <div class="model-name">{display_name}</div>'
        f'      <div class="model-arch">{architecture}</div>'
        f'    </div>'
        f'    <span class="{badge_cls}">{html.escape(side_role)} · Awaiting Weights</span>'
        f'  </div>'
        f'  <div class="standby-box">'
        f'    <div class="standby-title">Model Slot Ready · No Fabricated Predictions</div>'
        f'    <p class="standby-text">{reason}</p>'
        f'    <ul class="arch-list">'
        f'      <li>Drop weights as <span class="code-pill">backend/model/vit_0.keras..vit_4.keras</span> or <span class="code-pill">vit.keras</span>.</li>'
        f'      <li>Backend automatically runs <span class="code-pill">model(batch, training=False)</span> and computes identical confidence, Top-2 margin, and entropy diagnostics.</li>'
        f'      <li>Inductive bias comparison: <strong>Global Self-Attention</strong> (ViT) vs <strong>Local Depthwise Convolutions</strong> (EfficientNetB0).</li>'
        f'    </ul>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.button(f"Scan & Reload {display_name} Weights", key=f"reload_{model_key}", width="stretch"):
        with st.spinner("Scanning backend/model/ and warming up weights..."):
            _reloaded, err = reload_models(api_base)
            if err:
                st.error(err)
            else:
                st.session_state.pop("last_comparison", None)
                st.rerun()


def main() -> None:
    if "api_base" not in st.session_state:
        st.session_state["api_base"] = DEFAULT_API_BASE

    with st.sidebar:
        st.subheader("Workbench & Diagnostics")
        api_base_input = st.text_input(
            "FastAPI Backend URL",
            value=st.session_state["api_base"],
            help="Local or remote FastAPI base URL.",
        ).strip()
        if api_base_input:
            st.session_state["api_base"] = api_base_input

        api_base = st.session_state["api_base"]
        if st.button("Reload Models from Disk", width="stretch"):
            with st.spinner("Reloading models..."):
                _data, err = reload_models(api_base)
                if err:
                    st.error(err)
                else:
                    st.session_state.pop("last_comparison", None)
                    st.success("Model registry reloaded.")
                    st.rerun()

        st.divider()
        st.caption("Architecture Contrast")
        st.markdown(
            "- **Left · EfficientNetB0 (CNN)**: Compound-scaled depthwise separable convolutions with strong local texture & edge bias.\n"
            "- **Right · Vision Transformer (ViT)**: Patch embeddings + global multi-head self-attention for holistic shape & spatial context."
        )

    api_base = st.session_state["api_base"]
    health_data, health_err = fetch_health(api_base)
    models_data, _models_err = fetch_models(api_base)
    registry_models = (models_data or {}).get("models", {})
    classes = (health_data or {}).get("classes") or CLASS_LABELS

    # Build header status pills
    if health_data is None:
        pills_html = (
            '<span class="status-pill pill-offline">'
            '<span class="pill-dot dot-red"></span>API Offline</span>'
        )
    else:
        ef_info = registry_models.get("efficientnet_b0", {})
        vit_info = registry_models.get("vit", {})
        ef_ready = "efficientnet_b0" in health_data.get("ready_models", [])
        vit_ready = "vit" in health_data.get("ready_models", [])
        ef_folds = ef_info.get("fold_count", 0)
        vit_folds = vit_info.get("fold_count", 0)

        ef_pill = (
            f'<span class="status-pill pill-ready"><span class="pill-dot dot-emerald"></span>'
            f'EfficientNetB0 · Ready ({ef_folds} folds)</span>'
            if ef_ready
            else '<span class="status-pill pill-offline"><span class="pill-dot dot-red"></span>EfficientNetB0 · Unavailable</span>'
        )
        vit_pill = (
            f'<span class="status-pill pill-ready"><span class="pill-dot dot-indigo"></span>'
            f'ViT · Ready ({vit_folds} folds)</span>'
            if vit_ready
            else '<span class="status-pill pill-waiting"><span class="pill-dot dot-indigo"></span>ViT · Awaiting Weights</span>'
        )
        pills_html = ef_pill + vit_pill

    st.markdown(
        f'<div class="arena-header">'
        f'  <div>'
        f'    <h1 class="arena-title">EfficientNetB0 vs ViT</h1>'
        f'    <p class="arena-subtitle">6-Class Vegetable Classification · Head-to-Head CNN vs Transformer Benchmark</p>'
        f'  </div>'
        f'  <div class="status-cluster">{pills_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if health_err:
        st.error(
            f"{health_err} Run `uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000` to start the API."
        )

    with st.sidebar:
        if registry_models:
            st.divider()
            st.caption("Model Registry Status")
            for key, info in registry_models.items():
                st.write(
                    f"**{info['display_name']}** (`{key}`): `{info['status']}` · "
                    f"{info['fold_count']} weight file(s)"
                )

    # Compact Upload & Inspection Row
    col_upload, col_preview = st.columns([5, 7], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload vegetable specimen",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPEG, PNG, WEBP up to 10 MB (32–8192 px).",
        )
        max_mb = (health_data or {}).get("max_upload_mb", 10)
        st.caption(
            f"Validation pipeline: JPEG / PNG / WEBP · Max {max_mb:g} MB · 32–8192 px · EXIF auto-corrected"
        )

        current_hash: str | None = None
        file_bytes: bytes = b""
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            current_hash = hashlib.sha256(file_bytes).hexdigest()
            if st.session_state.get("last_image_hash") != current_hash:
                st.session_state["last_image_hash"] = current_hash
                st.session_state.pop("last_comparison", None)
                st.session_state.pop("last_error", None)
        else:
            st.session_state.pop("last_image_hash", None)
            st.session_state.pop("last_comparison", None)
            st.session_state.pop("last_error", None)

        run_disabled = uploaded_file is None or health_data is None
        if st.button(
            "Run Head-to-Head Comparison",
            type="primary",
            disabled=run_disabled,
            width="stretch",
        ):
            with st.spinner("Validating image and running model inference..."):
                result_payload, err = run_comparison(
                    api_base=api_base,
                    filename=uploaded_file.name,
                    file_bytes=file_bytes,
                    content_type=uploaded_file.type,
                )
            if err:
                st.session_state["last_error"] = err
                st.session_state.pop("last_comparison", None)
            else:
                st.session_state.pop("last_error", None)
                st.session_state["last_comparison"] = result_payload

    comparison_payload: dict[str, Any] | None = st.session_state.get("last_comparison")

    with col_preview:
        if uploaded_file is not None:
            preview_uri, local_meta = _build_preview_data_uri(file_bytes)
            server_img_meta = (comparison_payload or {}).get("image")

            if preview_uri:
                preview_html = f'<div class="preview-frame"><img src="{preview_uri}" alt="Uploaded specimen"/></div>'
            else:
                preview_html = (
                    '<div class="preview-frame">'
                    '<span style="color:#94A3B8;font-size:0.86rem;">Local preview unavailable — click Run to validate on backend.</span>'
                    '</div>'
                )

            fmt = (server_img_meta or local_meta or {}).get("format", "—")
            w = (server_img_meta or local_meta or {}).get("width", "—")
            h = (server_img_meta or local_meta or {}).get("height", "—")
            size_kb = len(file_bytes) / 1024.0
            size_label = f"{size_kb / 1024.0:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"

            ef_pred = ((comparison_payload or {}).get("models", {}).get("efficientnet_b0") or {}).get("predicted_label")
            vit_pred = ((comparison_payload or {}).get("models", {}).get("vit") or {}).get("predicted_label")

            chips_parts: list[str] = []
            for label in classes:
                esc_label = html.escape(label)
                if ef_pred == label and vit_pred == label:
                    chips_parts.append(f'<span class="class-chip class-chip-both">{esc_label} · Both</span>')
                elif ef_pred == label:
                    chips_parts.append(f'<span class="class-chip class-chip-ef">{esc_label} · EF-B0</span>')
                elif vit_pred == label:
                    chips_parts.append(f'<span class="class-chip class-chip-vit">{esc_label} · ViT</span>')
                else:
                    chips_parts.append(f'<span class="class-chip">{esc_label}</span>')

            st.markdown(
                f'<div class="surface-card">'
                f'  {preview_html}'
                f'  <div class="meta-row">'
                f'    <span class="meta-chip">{html.escape(uploaded_file.name)}</span>'
                f'    <span class="meta-chip">{html.escape(str(fmt))}</span>'
                f'    <span class="meta-chip">{html.escape(str(w))} × {html.escape(str(h))} px</span>'
                f'    <span class="meta-chip">{html.escape(size_label)}</span>'
                f'  </div>'
                f'  <div class="class-grid">{"".join(chips_parts)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            chips_html = "".join(f'<span class="class-chip">{html.escape(c)}</span>' for c in classes)
            st.markdown(
                f'<div class="surface-card">'
                f'  <div class="preview-frame">'
                f'    <div style="text-align:center;color:#94A3B8;padding:1rem;">'
                f'      <div style="font-weight:600;color:#E2E8F0;margin-bottom:0.25rem;">Specimen Inspection Frame</div>'
                f'      <div style="font-size:0.84rem;">Upload an image on the left to inspect metadata and run both models side by side.</div>'
                f'    </div>'
                f'  </div>'
                f'  <div class="class-grid">{chips_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    # Verdict & Uncertainty Banners
    if comparison_payload is not None:
        comp = comparison_payload.get("comparison", {})
        ef_res = comparison_payload["models"]["efficientnet_b0"]
        vit_res = comparison_payload["models"]["vit"]

        if comp.get("both_available"):
            agree = comp.get("models_agree")
            higher = comp.get("higher_confidence_model")
            faster = comp.get("faster_model")
            conf_delta = abs(float(comp.get("confidence_delta") or 0.0)) * 100.0
            lat_delta = abs(float(comp.get("latency_delta_ms") or 0.0))

            banner_cls = "verdict-agree" if agree else "verdict-disagree"
            verdict_title = (
                f"Models Agree: {html.escape(str(ef_res.get('predicted_label')))}"
                if agree
                else (
                    f"Models Disagree: EfficientNetB0 -> {html.escape(str(ef_res.get('predicted_label')))} "
                    f"vs ViT -> {html.escape(str(vit_res.get('predicted_label')))}"
                )
            )
            winner_name = "EfficientNetB0" if higher == "efficientnet_b0" else ("ViT" if higher == "vit" else "Tie")
            faster_name = "EfficientNetB0" if faster == "efficientnet_b0" else ("ViT" if faster == "vit" else "Tie")
            detail_line = (
                f"Confidence leader: <strong>{winner_name}</strong> (+{conf_delta:.1f}%) · "
                f"Faster: <strong>{faster_name}</strong> ({lat_delta:.1f} ms delta)"
            )
            st.markdown(
                f'<div class="verdict-banner {banner_cls}">'
                f'  <div style="font-weight:700;color:#F8FAFC;">{verdict_title}</div>'
                f'  <div style="font-size:0.86rem;color:#CBD5E1;">{detail_line}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif ef_res.get("status") == "ready":
            st.markdown(
                f'<div class="verdict-banner verdict-single">'
                f'  <div style="font-weight:700;color:#F8FAFC;">Active Benchmark: EfficientNetB0 ({html.escape(str(ef_res.get("predicted_label")))})</div>'
                f'  <div style="font-size:0.85rem;color:#CBD5E1;">ViT slot is on standby awaiting weights — comparison metrics activate automatically once ViT weights are loaded.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        all_warnings: list[str] = []
        for res in (ef_res, vit_res):
            if res.get("status") == "ready" and res.get("low_confidence_warning"):
                for msg in res.get("warning_reasons", []):
                    all_warnings.append(f"<strong>{html.escape(res['display_name'])}:</strong> {html.escape(msg)}")
        if all_warnings:
            st.markdown(
                f'<div class="caution-banner">'
                f'<strong>Uncertainty Caution:</strong> ' + " &nbsp;·&nbsp; ".join(all_warnings) +
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)

    # Main Left vs Right Arena
    col_left, col_right = st.columns(2, gap="large")

    ef_model_info = registry_models.get("efficientnet_b0")
    vit_model_info = registry_models.get("vit")
    ef_result = (comparison_payload or {}).get("models", {}).get("efficientnet_b0")
    vit_result = (comparison_payload or {}).get("models", {}).get("vit")

    with col_left:
        if ef_result is not None and ef_result.get("status") == "ready":
            _render_ready_model_card(
                result=ef_result,
                side_role="Left · CNN",
                card_accent_cls="model-card-ef",
                badge_cls="badge-ef",
                bar_fill_cls="prob-fill-ef",
            )
        elif ef_model_info and ef_model_info.get("status") == "unavailable":
            _render_standby_model_card(
                model_key="efficientnet_b0",
                model_info=ef_model_info,
                result=ef_result,
                side_role="Left · CNN",
                card_accent_cls="model-card-ef",
                badge_cls="badge-ef",
                api_base=api_base,
            )
        else:
            fold_count = (ef_model_info or {}).get("fold_count", 5)
            st.markdown(
                f'<div class="model-card model-card-ef">'
                f'  <div class="model-card-head">'
                f'    <div>'
                f'      <div class="model-name">EfficientNetB0</div>'
                f'      <div class="model-arch">Convolutional Neural Network (CNN)</div>'
                f'    </div>'
                f'    <span class="badge-ef">Left · CNN · {fold_count}-Fold Ready</span>'
                f'  </div>'
                f'  <div class="hero-prediction">'
                f'    <div class="hero-kicker">Ready for Inference</div>'
                f'    <div class="hero-label" style="font-size:1.18rem;color:#CBD5E1;">Awaiting Specimen Upload</div>'
                f'    <div class="hero-sub">Upload an image and click Run Head-to-Head Comparison to inspect per-class probabilities and fold consensus.</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_right:
        if vit_result is not None and vit_result.get("status") == "ready":
            _render_ready_model_card(
                result=vit_result,
                side_role="Right · ViT",
                card_accent_cls="model-card-vit",
                badge_cls="badge-vit",
                bar_fill_cls="prob-fill-vit",
            )
        elif vit_model_info and vit_model_info.get("status") == "ready":
            fold_count = vit_model_info.get("fold_count", 1)
            st.markdown(
                f'<div class="model-card model-card-vit">'
                f'  <div class="model-card-head">'
                f'    <div>'
                f'      <div class="model-name">Vision Transformer (ViT)</div>'
                f'      <div class="model-arch">Transformer Self-Attention over Image Patches</div>'
                f'    </div>'
                f'    <span class="badge-vit">Right · ViT · {fold_count} Ready</span>'
                f'  </div>'
                f'  <div class="hero-prediction">'
                f'    <div class="hero-kicker">Ready for Inference</div>'
                f'    <div class="hero-label" style="font-size:1.18rem;color:#CBD5E1;">Awaiting Specimen Upload</div>'
                f'    <div class="hero-sub">Upload an image and click Run Head-to-Head Comparison to compare self-attention predictions against EfficientNetB0.</div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            _render_standby_model_card(
                model_key="vit",
                model_info=vit_model_info,
                result=vit_result,
                side_role="Right · ViT",
                card_accent_cls="model-card-vit",
                badge_cls="badge-vit",
                api_base=api_base,
            )

    with st.sidebar:
        if comparison_payload is not None:
            st.divider()
            with st.expander("Raw JSON API Payload", expanded=False):
                st.json(comparison_payload)


if __name__ == "__main__":
    main()
