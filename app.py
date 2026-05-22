import tempfile
from collections import defaultdict

import cv2
import streamlit as st
from ultralytics import YOLO


@st.cache_resource(show_spinner="Loading model...")
def load_model(name):
    return YOLO(name)


def render_sidebar():
    st.sidebar.header("Source")
    source_type = st.sidebar.radio(
        "Input",
        ["Upload video", "Webcam", "Phone stream URL"],
        label_visibility="collapsed",
    )

    uploaded = cam_index = stream_url = None
    if source_type == "Upload video":
        uploaded = st.sidebar.file_uploader("Video file", type=["mp4", "mov", "avi"])
    elif source_type == "Webcam":
        cam_index = st.sidebar.number_input(
            "Camera index", min_value=0, max_value=5, value=1, step=1,
        )
    else:
        stream_url = st.sidebar.text_input(
            "Phone stream URL", "http://192.168.1.42:8080/video",
        )

    st.sidebar.header("Detector")
    model_name = st.sidebar.selectbox(
        "Model size (bigger is more accurate but slower)",
        [
            "yolov8n-oiv7.pt",
            "yolov8s-oiv7.pt",
            "yolov8m-oiv7.pt",
            "yolov8l-oiv7.pt",
            "yolov8x-oiv7.pt",
        ],
        index=2,
    )

    conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)

    st.sidebar.header("Controls")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("Start", use_container_width=True):
        st.session_state.running = True
    if c2.button("Stop", use_container_width=True):
        st.session_state.running = False
    
    if st.sidebar.button("Reset counts", use_container_width=True):
        st.session_state.unique_ids_per_class = defaultdict(set)

    return {
        "source_type": source_type,
        "uploaded": uploaded,
        "cam_index": cam_index,
        "stream_url": stream_url,
        "model_name": model_name,
        "conf": conf,
    }


def resolve_source(cfg):
    if cfg["source_type"] == "Upload video":
        if cfg["uploaded"] is None:
            return None

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(cfg["uploaded"].read())
        return tfile.name
    if cfg["source_type"] == "Webcam":
        return int(cfg["cam_index"])
    return cfg["stream_url"]

def draw_boxes(frame, boxes, class_names):
    out = frame.copy()
    if boxes is None or len(boxes) == 0:
        return out
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        confidence = float(box.conf[0])

        track_id = int(box.id[0]) if box.id is not None else None
        x1, y1, x2, y2, = map(int, box.xyxy[0].tolist())

        color = (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        parts = [cls_name]
        if track_id is not None:
            parts.append(f"id:{track_id}")
        parts.append(f"{confidence:.2f}")
        label = " ".join(parts)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            out, label, (x1 +3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def render_current_frame_stats(slot, counts):
    if not counts:
        slot.markdown("**Detections**\n\n_(none in this frame)_")
        return
    lines = ["**Detections (current frame)**", ""]
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")
    slot.markdown("\n".join(lines))

def render_unique_stats(slot, unique_ids_per_class):
    if not unique_ids_per_class:
        slot.markdown("Unique counts:\n (none yet)")
        return
    lines = ["Unique objects seen"]
    for k, ids in sorted(unique_ids_per_class.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"{k}: {len(ids)}")
    slot.markdown("\n".join(lines))


def run_inference(cfg, source, frame_slot, current_slot, unique_slot):
    model = load_model(cfg["model_name"])
    current_counts = defaultdict(int)
    unique_ids_per_class = st.session_state.unique_ids_per_class
 
    for r in model.track(
        source=source,
        stream=True,
        conf=cfg["conf"],
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
    ):
        if not st.session_state.running:
            break

        drawn = draw_boxes(r.orig_img, r.boxes, model.names)
        frame_slot.image(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
        
        current_counts.clear()
        if r.boxes is not None:
            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])]
                current_counts[cls_name] += 1
                if box.id is not None:
                    unique_ids_per_class[cls_name].add(int(box.id[0]))
    
        render_current_frame_stats(current_slot, current_counts)
        render_unique_stats(unique_slot, unique_ids_per_class)


def main():
    st.set_page_config(page_title="Live Object Detection", layout="wide")
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("unique_ids_per_class", defaultdict(set))
    st.title("Live Object Detection and tracking")
    st.caption("Live object detection with track ids")

    cfg = render_sidebar()
    source = resolve_source(cfg)

    col_video, col_stats = st.columns([3, 1])
    frame_slot = col_video.empty()
    current_slot = col_stats.empty()
    unique_slot = col_stats.empty()

    if source is None:
        frame_slot.info("Pick a source and press Start")
        return

    if not st.session_state.running:
        frame_slot.info("Source is ready. Press Start")
        return

    try:
        run_inference(cfg, source, frame_slot, current_slot, unique_slot)
    except Exception as e:
        st.session_state.running = False
        st.error(f"Could not open source: {source}\n\n{e}")


if __name__ == "__main__":
    main()
