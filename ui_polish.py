from io import BytesIO

import qrcode
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

HERO_KEY = "hero"

_STYLE = """
<style>
.stApp { background-color: #0e1117; }
.block-container { padding-top: 2.4rem; }

iframe[title="streamlit_image_coordinates"] {
    border: 1px solid #2a2f3a;
    border-radius: 10px;
    background: #07090d;
}

div[data-testid="stColumn"] div[data-testid="stVerticalBlock"]:has(.hero-badge) {
    position: relative;
    gap: 0;
}
.hero-badge {
    position: absolute; top: 14px; right: 18px; z-index: 20;
    display: flex; align-items: center; gap: 6px;
    font: 600 12px/1 system-ui, sans-serif; letter-spacing: 1px; color: #ffffff;
    background: rgba(7, 9, 13, 0.55); padding: 5px 9px; border-radius: 6px;
}
.live-dot {
    width: 8px; height: 8px; border-radius: 50%; background: #ff3b3b;
    box-shadow: 0 0 0 0 rgba(255, 59, 59, 0.7);
    animation: live-pulse 1.4s infinite;
}
@keyframes live-pulse {
    0% { box-shadow: 0 0 0 0 rgba(255, 59, 59, 0.7); }
    70% { box-shadow: 0 0 0 8px rgba(255, 59, 59, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 59, 59, 0); }
}

.hero-shell {
    display: flex; align-items: center; justify-content: center; height: 620px;
    border: 1px solid #2a2f3a; border-radius: 10px; background: #07090d;
    color: #8b94a7; font-size: 18px;
}

.lock-card {
    border: 1px solid #2a2f3a; border-radius: 10px; background: #12161f;
    padding: 14px 16px;
}
.lock-head {
    display: flex; align-items: baseline; justify-content: space-between;
    border-bottom: 1px solid #232a36; padding-bottom: 8px; margin-bottom: 12px;
}
.lock-title { font-size: 15px; font-weight: 600; color: #e6e6e6; }
.lock-id { font: 600 12px ui-monospace, monospace; color: #ff6b6b; }
.lock-metric { margin: 12px 0; }
.lock-num { font: 700 30px ui-monospace, Menlo, monospace; color: #ffd54a; line-height: 1.1; }
.lock-num.dim { color: #59617a; }
.lock-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #8b94a7; }
.lock-empty { color: #8b94a7; font-size: 14px; padding: 4px 0; }

.qr-url { font: 13px ui-monospace, monospace; color: #cfd6e4; text-align: center; word-break: break-all; margin-top: 10px; }
</style>
"""


def inject_css():
    """Apply the dark theme and component styling. Call once at startup."""
    st.markdown(_STYLE, unsafe_allow_html=True)


def make_qr_png(url):
    img = qrcode.make(url)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.dialog("Open on your phone")
def _qr_dialog(qr_url):
    if not qr_url:
        st.write("Set a phone stream URL in the sidebar first, then reopen this.")
        return
    _, mid, _ = st.columns([1, 2, 1])
    mid.image(make_qr_png(qr_url))
    st.markdown(f'<div class="qr-url">{qr_url}</div>', unsafe_allow_html=True)
    if st.button("Close", use_container_width=True):
        st.rerun()


def phone_qr_button(qr_url):
    """Render the 'Open on your phone' button that opens the QR dialog."""
    _, slot = st.columns([3, 1])
    if slot.button("Open on your phone", use_container_width=True):
        _qr_dialog(qr_url)


def box_at_point(boxes, x, y):
    hit = None
    best = None
    for tid, x1, y1, x2, y2 in boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best:
                best = area
                hit = tid
    return hit


def render_hero_frame(slot, frame_rgb):
    """Show the annotated frame in the hero container and return any click."""
    if frame_rgb is None:
        slot.markdown(
            '<div class="hero-shell">Pick a source, then click Start</div>',
            unsafe_allow_html=True,
        )
        return None
    with slot.container():
        st.markdown(
            '<div class="hero-badge"><span class="live-dot"></span>LIVE</div>',
            unsafe_allow_html=True,
        )
        return streamlit_image_coordinates(
            frame_rgb, key=HERO_KEY, use_column_width="always", cursor="crosshair",
        )


def _card(body):
    return f'<div class="lock-card">{body}</div>'


def _head(title, tid):
    id_html = f'<span class="lock-id">id {tid}</span>' if tid is not None else ""
    return f'<div class="lock-head"><span class="lock-title">{title}</span>{id_html}</div>'


def _metric(label, value, dim=False):
    cls = "lock-num dim" if dim else "lock-num"
    return (
        f'<div class="lock-metric"><div class="{cls}">{value}</div>'
        f'<div class="lock-label">{label}</div></div>'
    )


def render_locked_panel(slot, info, lock_id):
    """Render the locked target card with large live metrics for speed and path."""
    if lock_id is None:
        body = _head("Locked target", None) + (
            '<div class="lock-empty">No target locked. '
            "Click an object in the video to follow it.</div>"
        )
        slot.markdown(_card(body), unsafe_allow_html=True)
        return
    if info is None:
        body = (
            _head("Locked target", lock_id)
            + _metric("Speed (px/s)", "--", dim=True)
            + _metric("Path (px)", "--", dim=True)
            + '<div class="lock-empty">Not visible in this frame.</div>'
        )
        slot.markdown(_card(body), unsafe_allow_html=True)
        return
    body = (
        _head(info["class"], info["id"])
        + _metric("Speed (px/s)", f"{info['pix_speed']:.0f}")
        + _metric("Path (px)", f"{info['path_px']:.0f}")
    )
    slot.markdown(_card(body), unsafe_allow_html=True)
