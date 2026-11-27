import base64
import hashlib
import html
from io import BytesIO
import os
from typing import Any

from PIL import Image, ImageOps
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
CLASSES = ["Bean", "Bitter melon", "Cucumber", "Green Chili", "Ladies finger", "Pointed gourd"]
SESSION_RESULT_KEY = "arena_result_v2"
SESSION_HASH_KEY = "arena_image_hash_v2"

DARK_ARENA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
[data-testid="stAppViewContainer"] { background-color: #0B0F19; color: #F1F5F9; }
[data-testid="stHeader"] { background-color: rgba(11, 15, 25, 0.85); }
[data-testid="stFileUploaderDropzone"] { background-color: #111827; border: 1px dashed #334155; border-radius: 12px; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1280px; padding-top: 1.35rem; padding-bottom: 2.2rem; }

.arena-header {
    display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;
    padding-bottom: 0.9rem; margin-bottom: 1.15rem; border-bottom: 1px solid #1E293B; flex-wrap: wrap;
}
.arena-title { margin: 0; font-size: 1.7rem; font-weight: 700; letter-spacing: -0.025em; color: #F8FAFC; }
.arena-subtitle { margin: 0.25rem 0 0 0; font-size: 0.9rem; color: #94A3B8; }
.status-cluster { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.status-pill {
    display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.3rem 0.72rem;
    border-radius: 9999px; font-size: 0.78rem; font-weight: 600; border: 1px solid transparent;
}
.pill-dino { background: rgba(99, 102, 241, 0.14); color: #C7D2FE; border-color: rgba(99, 102, 241, 0.35); }
.pill-ef { background: rgba(16, 185, 129, 0.12); color: #6EE7B7; border-color: rgba(16, 185, 129, 0.32); }
.pill-warn { background: rgba(245, 158, 11, 0.14); color: #FDE68A; border-color: rgba(245, 158, 11, 0.35); }
.pill-offline { background: rgba(239, 68, 68, 0.14); color: #FCA5A5; border-color: rgba(239, 68, 68, 0.35); }
.pill-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }

.surface-card {
    background: #111827; border: 1px solid #1F2937; border-radius: 14px;
    padding: 1rem 1.15rem; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}
.preview-frame {
    background: #0B1220; border: 1px solid #1E293B; border-radius: 10px;
    padding: 0.5rem; display: flex; justify-content: center; align-items: center; min-height: 170px;
}
.preview-frame img { max-height: 235px; max-width: 100%; width: auto; height: auto; object-fit: contain; border-radius: 6px; }
.meta-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.7rem; }
.meta-chip {
    font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; color: #CBD5E1;
    background: #1E293B; border: 1px solid #334155; border-radius: 6px; padding: 0.18rem 0.5rem;
}
.class-grid { display: flex; flex-wrap: wrap; gap: 0.38rem; margin-top: 0.6rem; }
.class-chip {
    font-size: 0.75rem; font-weight: 500; color: #94A3B8; background: #0F172A;
    border: 1px solid #1E293B; border-radius: 9999px; padding: 0.2rem 0.6rem;
}
.chip-dino { background: rgba(99, 102, 241, 0.18); color: #C7D2FE; border-color: rgba(99, 102, 241, 0.45); font-weight: 600; }
.chip-ef { background: rgba(16, 185, 129, 0.15); color: #6EE7B7; border-color: rgba(16, 185, 129, 0.45); font-weight: 600; }
.chip-both { background: #F8FAFC; color: #0B0F19; border-color: #F8FAFC; font-weight: 700; }

.verdict-banner {
    border-radius: 12px; padding: 0.85rem 1.1rem; margin: 0.9rem 0 1rem 0;
    border: 1px solid #1F2937; display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;
}
.verdict-agree { border-left: 4px solid #10B981; background: rgba(16, 185, 129, 0.10); border-color: rgba(16, 185, 129, 0.30); }
.verdict-disagree { border-left: 4px solid #F59E0B; background: rgba(245, 158, 11, 0.10); border-color: rgba(245, 158, 11, 0.30); }

.model-card {
    background: #111827; border: 1px solid #1F2937; border-radius: 14px;
    padding: 1.2rem 1.25rem; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}
.card-dino { border-top: 4px solid #6366F1; }
.card-ef { border-top: 4px solid #10B981; }
.model-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.75rem; margin-bottom: 0.85rem; }
.model-name { font-size: 1.12rem; font-weight: 700; color: #F8FAFC; margin: 0; }
.model-arch { font-size: 0.8rem; color: #94A3B8; margin-top: 0.15rem; }
.badge-dino {
    background: rgba(99, 102, 241, 0.16); color: #C7D2FE; border: 1px solid rgba(99, 102, 241, 0.35);
    font-size: 0.72rem; font-weight: 700; padding: 0.22rem 0.6rem; border-radius: 9999px; white-space: nowrap;
}
.badge-ef {
    background: rgba(16, 185, 129, 0.14); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.35);
    font-size: 0.72rem; font-weight: 700; padding: 0.22rem 0.6rem; border-radius: 9999px; white-space: nowrap;
}
.hero-box { background: #0B1220; border: 1px solid #1E293B; border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.9rem; }
.hero-kicker { font-size: 0.71rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; }
.hero-label { font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin: 0.15rem 0 0.22rem 0; line-height: 1.15; }
.hero-sub { font-size: 0.81rem; color: #CBD5E1; font-family: 'JetBrains Mono', monospace; }

.metric-strip { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.55rem; margin-bottom: 0.95rem; }
.metric-tile { background: #0B1220; border: 1px solid #1E293B; border-radius: 9px; padding: 0.55rem 0.75rem; }
.metric-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #94A3B8; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 600; color: #F8FAFC; margin-top: 0.12rem; }
.caution-note {
    margin-bottom: 0.85rem; padding: 0.6rem 0.85rem; border-radius: 8px; font-size: 0.82rem;
    background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.32); color: #FDE68A;
}

.prob-title { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 0.5rem; }
.prob-row { margin-bottom: 0.5rem; }
.prob-row-top { display: flex; justify-content: space-between; font-size: 0.83rem; margin-bottom: 0.2rem; }
.prob-label { color: #CBD5E1; font-weight: 500; }
.prob-label-top { color: #F8FAFC; font-weight: 700; }
.prob-num { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #CBD5E1; }
.prob-track { width: 100%; height: 8px; background: #1E293B; border-radius: 9999px; overflow: hidden; }
.fill-dino { height: 100%; border-radius: 9999px; background: linear-gradient(90deg, #818CF8, #4F46E5); }
.fill-ef { height: 100%; border-radius: 9999px; background: linear-gradient(90deg, #10B981, #059669); }
.fill-muted { height: 100%; border-radius: 9999px; background: #475569; }
</style>
"""


def _is_valid_prediction(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("label"), str)
        and isinstance(payload.get("confidence"), (int, float))
        and isinstance(payload.get("inference_ms"), (int, float))
        and isinstance(payload.get("probabilities"), dict)
    )


class ClassifierClient:
    """HTTP adapter for the FastAPI DINOv2 & EfficientNetB0 backend."""

    def __init__(self, base_url: str = API_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def check_backend(self) -> dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=4)
            if not resp.ok:
                return {"state": "offline"}
            data = resp.json()
        except (requests.RequestException, ValueError):
            return {"state": "offline"}

        models = data.get("models") if isinstance(data, dict) else None
        if isinstance(models, dict) and "dinov2" in models and "efb0" in models:
            return {"state": "ready", "data": data}
        return {"state": "legacy", "raw": data}

    def _post_image(
        self,
        endpoint: str,
        file_bytes: bytes,
        filename: str,
        content_type: str | None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        files = {"file": (filename or "image.jpg", file_bytes, content_type or "application/octet-stream")}
        try:
            resp = requests.post(f"{self.base_url}{endpoint}", files=files, params=params, timeout=45)
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach API at {self.base_url}: {exc}") from exc

        if not resp.ok:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise RuntimeError(f"API error ({resp.status_code}): {detail}")

        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError("Invalid JSON response from backend.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Unexpected response format from backend.")
        return payload

    def predict(self, file_bytes: bytes, filename: str, content_type: str | None, model: str) -> dict[str, Any]:
        payload = self._post_image("/predict", file_bytes, filename, content_type, params={"model": model})
        if not _is_valid_prediction(payload):
            raise RuntimeError("Invalid prediction payload from backend. Please restart the FastAPI server.")
        return payload

    def compare(self, file_bytes: bytes, filename: str, content_type: str | None) -> dict[str, Any]:
        payload = self._post_image("/predict/compare", file_bytes, filename, content_type)
        results = payload.get("results")
        if not isinstance(results, dict) or not results or not all(_is_valid_prediction(v) for v in results.values()):
            raise RuntimeError(
                "Outdated backend server detected on port 8000. Stop the old uvicorn terminal (Ctrl+C) "
                "and restart: uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000"
            )
        return payload


def _build_preview_uri(file_bytes: bytes) -> tuple[str | None, dict[str, Any] | None]:
    try:
        with Image.open(BytesIO(file_bytes)) as img:
            fmt = (img.format or "IMG").upper()
            transposed = ImageOps.exif_transpose(img)
            width, height = transposed.size
            rgb = transposed.convert("RGB")
            rgb.thumbnail((900, 900))
            buf = BytesIO()
            rgb.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}", {"format": fmt, "width": width, "height": height}
    except Exception:
        return None, None


def _get_saved_results() -> dict[str, dict[str, Any]] | None:
    saved = st.session_state.get(SESSION_RESULT_KEY)
    if not isinstance(saved, dict) or not saved:
        st.session_state.pop(SESSION_RESULT_KEY, None)
        return None
    if not all(k in ("dinov2", "efb0") and _is_valid_prediction(v) for k, v in saved.items()):
        st.session_state.pop(SESSION_RESULT_KEY, None)
        return None
    return saved


def _render_prob_bars(probabilities: dict[str, float], fill_cls: str) -> str:
    sorted_items = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)
    rows: list[str] = []
    for rank, (cls_name, prob_raw) in enumerate(sorted_items):
        pct = float(prob_raw) * 100.0
        bar_w = max(1.5, min(100.0, pct)) if pct > 0.0 else 0.0
        lbl_cls = "prob-label prob-label-top" if rank == 0 else "prob-label"
        bar_cls = fill_cls if rank == 0 else "fill-muted"
        rows.append(
            f'<div class="prob-row">'
            f'  <div class="prob-row-top"><span class="{lbl_cls}">{html.escape(str(cls_name))}</span>'
            f'  <span class="prob-num">{pct:.1f}%</span></div>'
            f'  <div class="prob-track"><div class="{bar_cls}" style="width: {bar_w:.1f}%;"></div></div>'
            f'</div>'
        )
    return "".join(rows)


def _render_model_card(
    title: str,
    subtitle: str,
    badge: str,
    card_cls: str,
    badge_cls: str,
    fill_cls: str,
    result: dict[str, Any] | None,
) -> None:
    if result is None:
        body = (
            '<div class="hero-box">'
            '  <div class="hero-kicker">Ready for Inference</div>'
            '  <div class="hero-label" style="font-size:1.15rem;color:#CBD5E1;">Awaiting Prediction</div>'
            '  <div class="hero-sub">Upload an image and click Predict Vegetable to view probabilities.</div>'
            '</div>'
        )
    else:
        label = html.escape(str(result["label"]))
        conf_pct = float(result["confidence"]) * 100.0
        ms = float(result["inference_ms"])
        caution = '<div class="caution-note">Low confidence prediction (&lt; 60%).</div>' if result.get("low_confidence") else ""
        bars = _render_prob_bars(result["probabilities"], fill_cls)
        body = (
            f'<div class="hero-box">'
            f'  <div class="hero-kicker">Top-1 Prediction</div>'
            f'  <div class="hero-label">{label}</div>'
            f'  <div class="hero-sub">5-Fold Mean Ensemble · {ms:.1f} ms</div>'
            f'</div>'
            f'<div class="metric-strip">'
            f'  <div class="metric-tile"><div class="metric-label">Confidence</div><div class="metric-value">{conf_pct:.2f}%</div></div>'
            f'  <div class="metric-tile"><div class="metric-label">Inference Time</div><div class="metric-value">{ms:.1f} ms</div></div>'
            f'</div>'
            f'{caution}'
            f'<div class="prob-title">Class Probabilities</div>'
            f'{bars}'
        )

    st.markdown(
        f'<div class="model-card {card_cls}">'
        f'  <div class="model-head">'
        f'    <div><div class="model-name">{html.escape(title)}</div><div class="model-arch">{html.escape(subtitle)}</div></div>'
        f'    <span class="{badge_cls}">{html.escape(badge)}</span>'
        f'  </div>'
        f'  {body}'
        f'</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="DINOv2 vs EfficientNetB0 · Vegetable Classifier", page_icon="🥬", layout="wide")
    st.markdown(DARK_ARENA_CSS, unsafe_allow_html=True)

    client = ClassifierClient()
    status = client.check_backend()
    state = status["state"]

    if state == "ready":
        models_info = status["data"].get("models", {})
        dino_folds = int(models_info.get("dinov2", 0))
        ef_folds = int(models_info.get("efb0", 0))
        pills_html = (
            f'<span class="status-pill pill-dino"><span class="pill-dot" style="background:#6366F1;"></span>DINOv2 · {dino_folds} Folds Ready</span>'
            f'<span class="status-pill pill-ef"><span class="pill-dot" style="background:#10B981;"></span>EfficientNetB0 · {ef_folds} Folds Ready</span>'
        )
        classes = status["data"].get("classes") or CLASSES
    elif state == "legacy":
        dino_folds = ef_folds = 5
        pills_html = '<span class="status-pill pill-warn"><span class="pill-dot" style="background:#F59E0B;"></span>Outdated Backend on :8000</span>'
        classes = CLASSES
    else:
        dino_folds = ef_folds = 5
        pills_html = '<span class="status-pill pill-offline"><span class="pill-dot" style="background:#EF4444;"></span>API Offline</span>'
        classes = CLASSES

    st.markdown(
        f'<div class="arena-header">'
        f'  <div>'
        f'    <h1 class="arena-title">DINOv2 vs EfficientNetB0</h1>'
        f'    <p class="arena-subtitle">6-Class Green Vegetable Classification · 5-Fold Ensemble Arena</p>'
        f'  </div>'
        f'  <div class="status-cluster">{pills_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if state == "legacy":
        st.warning(
            "An older FastAPI server is still running on port 8000. "
            "Stop it with `Ctrl+C` in that terminal and restart: `uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000`"
        )
    elif state == "offline":
        st.error(
            f"Backend offline at `{client.base_url}`. "
            "Start it with: `uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000`"
        )

    col_upload, col_preview = st.columns([5, 7], gap="large")
    with col_upload:
        uploaded = st.file_uploader("Upload vegetable image", type=["jpg", "jpeg", "png", "webp"])
        mode = st.radio("Inference Mode", ["Compare Both", "DINOv2 Only", "EfficientNetB0 Only"], horizontal=True)

        file_bytes = uploaded.getvalue() if uploaded is not None else b""
        if uploaded is not None:
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            if st.session_state.get(SESSION_HASH_KEY) != file_hash:
                st.session_state[SESSION_HASH_KEY] = file_hash
                st.session_state.pop(SESSION_RESULT_KEY, None)
        else:
            st.session_state.pop(SESSION_HASH_KEY, None)
            st.session_state.pop(SESSION_RESULT_KEY, None)

        run_clicked = st.button(
            "Predict Vegetable",
            type="primary",
            width="stretch",
            disabled=(state != "ready" or uploaded is None),
        )

    if run_clicked and uploaded is not None:
        with st.spinner("Running 5-fold ensemble inference..."):
            try:
                if mode == "Compare Both":
                    cmp_payload = client.compare(file_bytes, uploaded.name, uploaded.type)
                    st.session_state[SESSION_RESULT_KEY] = cmp_payload["results"]
                elif mode == "DINOv2 Only":
                    res = client.predict(file_bytes, uploaded.name, uploaded.type, "dinov2")
                    st.session_state[SESSION_RESULT_KEY] = {"dinov2": res}
                else:
                    res = client.predict(file_bytes, uploaded.name, uploaded.type, "efb0")
                    st.session_state[SESSION_RESULT_KEY] = {"efb0": res}
            except RuntimeError as exc:
                st.error(str(exc))
                st.session_state.pop(SESSION_RESULT_KEY, None)

    saved = _get_saved_results()
    dino_res = (saved or {}).get("dinov2")
    ef_res = (saved or {}).get("efb0")

    with col_preview:
        dino_pred = dino_res["label"] if dino_res else None
        ef_pred = ef_res["label"] if ef_res else None
        chips: list[str] = []
        for cls_name in classes:
            esc = html.escape(str(cls_name))
            if dino_pred == cls_name and ef_pred == cls_name:
                chips.append(f'<span class="class-chip chip-both">{esc} · Both</span>')
            elif dino_pred == cls_name:
                chips.append(f'<span class="class-chip chip-dino">{esc} · DINOv2</span>')
            elif ef_pred == cls_name:
                chips.append(f'<span class="class-chip chip-ef">{esc} · EF-B0</span>')
            else:
                chips.append(f'<span class="class-chip">{esc}</span>')

        if uploaded is not None:
            uri, meta = _build_preview_uri(file_bytes)
            frame = f'<div class="preview-frame"><img src="{uri}" alt="Uploaded specimen"/></div>' if uri else '<div class="preview-frame"><span style="color:#94A3B8;">Preview unavailable</span></div>'
            fmt = (meta or {}).get("format", "—")
            w = (meta or {}).get("width", "—")
            h = (meta or {}).get("height", "—")
            kb = len(file_bytes) / 1024.0
            size_txt = f"{kb / 1024.0:.2f} MB" if kb >= 1024 else f"{kb:.1f} KB"
            meta_html = (
                f'<div class="meta-row">'
                f'  <span class="meta-chip">{html.escape(uploaded.name)}</span>'
                f'  <span class="meta-chip">{html.escape(str(fmt))}</span>'
                f'  <span class="meta-chip">{html.escape(str(w))} × {html.escape(str(h))} px</span>'
                f'  <span class="meta-chip">{html.escape(size_txt)}</span>'
                f'</div>'
            )
        else:
            frame = (
                '<div class="preview-frame"><div style="text-align:center;color:#94A3B8;padding:1rem;">'
                '<div style="font-weight:600;color:#E2E8F0;margin-bottom:0.2rem;">Specimen Inspection Frame</div>'
                '<div style="font-size:0.84rem;">Upload a vegetable photo to compare DINOv2 and EfficientNetB0 side by side.</div>'
                '</div></div>'
            )
            meta_html = ""

        st.markdown(
            f'<div class="surface-card">{frame}{meta_html}<div class="class-grid">{"".join(chips)}</div></div>',
            unsafe_allow_html=True,
        )

    if dino_res and ef_res:
        agree = dino_res["label"] == ef_res["label"]
        banner_cls = "verdict-agree" if agree else "verdict-disagree"
        title_txt = (
            f"Models Agree: {html.escape(dino_res['label'])}"
            if agree
            else f"Models Disagree: DINOv2 -> {html.escape(dino_res['label'])} vs EfficientNetB0 -> {html.escape(ef_res['label'])}"
        )
        d_conf, e_conf = float(dino_res["confidence"]) * 100.0, float(ef_res["confidence"]) * 100.0
        d_ms, e_ms = float(dino_res["inference_ms"]), float(ef_res["inference_ms"])
        conf_leader = "DINOv2" if d_conf >= e_conf else "EfficientNetB0"
        faster = "DINOv2" if d_ms <= e_ms else "EfficientNetB0"
        st.markdown(
            f'<div class="verdict-banner {banner_cls}">'
            f'  <div style="font-weight:700;color:#F8FAFC;">{title_txt}</div>'
            f'  <div style="font-size:0.85rem;color:#CBD5E1;">Confidence leader: <strong>{conf_leader}</strong> (+{abs(d_conf - e_conf):.1f}%) · Faster: <strong>{faster}</strong> ({abs(d_ms - e_ms):.1f} ms delta)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2, gap="large")
    with col_left:
        _render_model_card(
            title="DINOv2",
            subtitle="Self-Supervised Vision Transformer (ViT-S/14)",
            badge=f"Left · DINOv2 · {dino_folds}-Fold Ensemble",
            card_cls="card-dino",
            badge_cls="badge-dino",
            fill_cls="fill-dino",
            result=dino_res,
        )
    with col_right:
        _render_model_card(
            title="EfficientNetB0",
            subtitle="Convolutional Neural Network (EfficientNetB0 v4)",
            badge=f"Right · EfficientNetB0 · {ef_folds}-Fold Ensemble",
            card_cls="card-ef",
            badge_cls="badge-ef",
            fill_cls="fill-ef",
            result=ef_res,
        )


if __name__ == "__main__":
    main()
