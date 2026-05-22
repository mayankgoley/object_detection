import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from collections import defaultdict

st.set_page_config(page_title="Live Perception", layout="wide")
st.title("Real Time Object Detection and Tracking")

# Sidebar controls
source_type = st.sidebar.radio("Source", ["Upload video", "Webcam", "Phone stream URL"])
conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.4)
model_name = st.sidebar.selectbox(
    "Model (bigger = more accurate, slower)",
    ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
)


@st.cache_resource
def load_model(name):
    return YOLO(name)


model = load_model(model_name)

# Resolve source
if source_type == "Upload video":
    uploaded = st.sidebar.file_uploader("Choose a video", type=["mp4", "mov", "avi"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        source = tfile.name
    else:
        source = None
elif source_type == "Webcam":
    source = int(
        st.sidebar.number_input("Camera index", min_value=0, max_value=5, value=1, step=1)
    )
else:
    source = st.sidebar.text_input("Phone stream URL", "http://192.168.1.42:8080/video")

# Start/Stop controls
if "running" not in st.session_state:
    st.session_state.running = False
if st.sidebar.button("Start"):
    st.session_state.running = True
if st.sidebar.button("Stop"):
    st.session_state.running = False

# Run inference if source is set
if source is not None and st.session_state.running:
    frame_placeholder = st.empty()
    stats_placeholder = st.sidebar.empty()
    counts = defaultdict(set)

    try:
        for r in model.track(source=source, stream=True, conf=conf, tracker="bytetrack.yaml"):
            frame = r.plot()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb)

            # Track unique IDs per class
            if r.boxes.id is not None:
                for cls, track_id in zip(r.boxes.cls.tolist(), r.boxes.id.tolist()):
                    counts[model.names[int(cls)]].add(int(track_id))

            # Show counts
            summary = "\n".join(f"**{k}**: {len(v)} unique" for k, v in counts.items())
            stats_placeholder.markdown(summary or "No detections yet")
    except Exception as e:
        st.session_state.running = False
        st.error(f"Could not open source: {source}\n\n{e}")
