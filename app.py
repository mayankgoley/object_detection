import os
import tempfile
from collections import defaultdict

import cv2
import streamlit as st
from ultralytics import YOLO

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:
    st.error(
        "Missing av, streamlit-webrtc, or streamlit-autorefresh. Launch with "
        "./run.sh so the app uses the project venv at /Users/goley/.venvs/object_detection."
    )
    st.stop()


ON_SPACES = "SPACE_ID" in os.environ


def ice_servers():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    turn_urls = os.environ.get("TURN_URLS")
    if turn_urls:
        servers.append({
            "urls": [u.strip() for u in turn_urls.split(",") if u.strip()],
            "username": os.environ.get("TURN_USERNAME", ""),
            "credential": os.environ.get("TURN_CREDENTIAL", ""),
        })
    return servers


@st.cache_resource(show_spinner="Loading model...")
def load_model(name):
    return YOLO(name)

def parse_classes(text):
    return [c.strip() for c in text.split(",") if c.strip()]


def render_sidebar():
    st.sidebar.header("Source")
    sources = ["Upload video", "Live webcam (browser)"]
    if not ON_SPACES:
        sources += ["Webcam (local)", "Phone stream URL"]
    source_type = st.sidebar.radio(
        "Input", sources, label_visibility="collapsed",
    )

    uploaded = cam_index = stream_url = None
    if source_type == "Upload video":
        uploaded = st.sidebar.file_uploader("Video file", type=["mp4", "mov", "avi"])
    elif source_type == "Webcam (local)":
        cam_index = st.sidebar.number_input(
            "Camera index", min_value=0, max_value=5, value=1, step=1,
        )
    elif source_type == "Phone stream URL":
        stream_url = st.sidebar.text_input(
            "Phone stream URL", "http://192.168.1.42:8080/video",
        )

    detector_mode = st.sidebar.radio(
        "Mode",
        [
            "Closed set (YOLOv8-OIv7)",
            "Fast (YOLO11-COCO)",
            "Open vocabulary (YOLO-World)",
        ],
        help=(
            "Closed set: 600 fixed classes from Open Images V7. Accurate but heavy. "
            "Fast: 80 COCO classes with YOLO11, much higher frame rate. "
            "Open vocab: type any class names, YOLO-World detects them. Slower."
        ),
    )

    custom_classes_text = ""
    if detector_mode == "Closed set (YOLOv8-OIv7)":
        model_name = st.sidebar.selectbox(
            "Model size (bigger means more accurate, slower)",
            [
                "yolov8n-oiv7.pt", "yolov8s-oiv7.pt", "yolov8m-oiv7.pt",
                "yolov8l-oiv7.pt", "yolov8x-oiv7.pt",
            ],
            index=0 if ON_SPACES else 2,
        )
    elif detector_mode == "Fast (YOLO11-COCO)":
        model_name = st.sidebar.selectbox(
            "Model size (bigger means more accurate, slower)",
            [
                "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
                "yolo11l.pt", "yolo11x.pt",
            ],
            index=0,
        )
    else:
        model_name = st.sidebar.selectbox(
            "Model size",
            ["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt"],
            index=0,
        )
        custom_classes_text = st.sidebar.text_area(
            "Classes (comma separated)",
            "person, closet, window, door, chair, table, bed, tv, laptop",
            help="Type any object names. YOLO-World will detect only these.",
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
        "detector_mode": detector_mode,
        "model_name": model_name,
        "custom_classes_text": custom_classes_text,
        "conf": conf,
    }


def resolve_source(cfg):
    if cfg["source_type"] == "Upload video":
        if cfg["uploaded"] is None:
            return None

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(cfg["uploaded"].read())
        return tfile.name
    if cfg["source_type"] == "Webcam (local)":
        return int(cfg["cam_index"])
    return cfg["stream_url"]

def draw_boxes(frame, boxes, class_names, box_color):
    out = frame.copy()
    if boxes is None or len(boxes) == 0:
        return out
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = class_names[cls_id]
        confidence = float(box.conf[0])

        track_id = int(box.id[0]) if box.id is not None else None
        x1, y1, x2, y2, = map(int, box.xyxy[0].tolist())

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
        parts = [cls_name]
        if track_id is not None:
            parts.append(f"id:{track_id}")
        parts.append(f"{confidence:.2f}")
        label = " ".join(parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1 + tw + 6, y1), box_color, -1)
        cv2.putText(
            out, label, (x1 +3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out

class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.conf = 0.4
        self.box_color = (0, 200, 0)
        self.current_counts = defaultdict(int)
        self.unique_ids_per_class = defaultdict(set)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        results = self.model.track(
            img, conf=self.conf, persist=True,
            tracker="bytetrack.yaml", verbose=False,
        )
        r = results[0]
        drawn = draw_boxes(img, r.boxes, self.model.names, self.box_color)
        counts = defaultdict(int)
        if r.boxes is not None:
            for box in r.boxes:
                cls_name = self.model.names[int(box.cls[0])]
                counts[cls_name] += 1
                if box.id is not None:
                    self.unique_ids_per_class[cls_name].add(int(box.id[0]))
        self.current_counts = counts
        return av.VideoFrame.from_ndarray(drawn, format="bgr24")

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
    if cfg["detector_mode"] == "Open vocabulary (YOLO-World)":
        classes = parse_classes(cfg["custom_classes_text"])
        if classes:
            model.set_classes(classes)
        box_color = (0, 165, 255)
    else:
        box_color = (0, 200, 0)

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

        drawn = draw_boxes(r.orig_img, r.boxes, model.names, box_color)
        frame_slot.image(
            cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), width="stretch",
        )
        
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
    st.markdown(
        "<style>.block-container { padding: 2rem 1.5rem 1rem; max-width: 100%; }</style>",
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("unique_ids_per_class", defaultdict(set))
    st.title("Live Object Detection and tracking")
    st.caption("Live object detection with track ids")

    cfg = render_sidebar()
    source = resolve_source(cfg)

    col_video, col_stats = st.columns([5, 1])

    if cfg["source_type"] == "Live webcam (browser)":
        with col_video:
            ctx = webrtc_streamer(
                key="yolo",
                video_processor_factory=YOLOProcessor,
                rtc_configuration=RTCConfiguration({"iceServers": ice_servers()}),
                media_stream_constraints={"video": True, "audio": False},
            )
        if ctx.video_processor:
            ctx.video_processor.model = load_model(cfg["model_name"])
            ctx.video_processor.conf = cfg["conf"]
            if cfg["detector_mode"] == "Open vocabulary (YOLO-World)":
                classes = parse_classes(cfg["custom_classes_text"])
                if classes:
                    ctx.video_processor.model.set_classes(classes)
                ctx.video_processor.box_color = (0, 165, 255)
            else:
                ctx.video_processor.box_color = (0, 200, 0)

        current_slot = col_stats.empty()
        unique_slot = col_stats.empty()
        if ctx.state.playing:
            st_autorefresh(interval=500, key="webrtc_stats")
            vp = ctx.video_processor
            if vp is not None:
                render_current_frame_stats(current_slot, vp.current_counts)
                render_unique_stats(unique_slot, vp.unique_ids_per_class)
        else:
            current_slot.info("Press Start above to begin detection.")
        return

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
