import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io
import pathlib

VOC_TO_YOLO_CLASSES = ["Block crack", "D00", "D10", "D20", "D40", "Repair"]
SEVERITY_MAP = {
    "Block crack": 3,
    "D40": 3,
    "D20": 2,
    "D10": 1,
    "D00": 1,
    "Repair": 0,
}
SEVERITY_LABEL = {0: "None/Repair", 1: "Low", 2: "Medium", 3: "High"}


def find_latest_run_best_model(root: pathlib.Path) -> pathlib.Path | None:
    candidates = list(root.rglob("best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def draw_boxes_on_image(image: np.ndarray, boxes):
    img = image.copy()
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = f"{VOC_TO_YOLO_CLASSES[cls]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def main():
    st.title("Road Damage Detection — Demo UI")
    st.write("Upload an image and the model will detect road damage and show severity and confidence.")

    app_root = pathlib.Path(__file__).resolve().parent
    default_model = app_root / "model" / "best.pt"
    latest_run_model = find_latest_run_best_model(app_root / "runs" / "detect" / "runs" / "train")
    if not default_model.exists():
        candidate_models = sorted((app_root / "model").glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidate_models:
            default_model = candidate_models[0]
        elif latest_run_model is not None:
            default_model = latest_run_model

    if not default_model.exists():
        st.error("No trained model file found. Place a .pt model under model/ or train the model first.")
        return

    model_path = str(default_model)
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to run prediction.")

    if uploaded is not None and 'model' not in st.session_state:
        try:
            model = YOLO(model_path)
            st.session_state['model'] = model
            st.success(f"Loaded model: {model_path}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return

    if uploaded and 'model' in st.session_state:
        image = Image.open(io.BytesIO(uploaded.read())).convert('RGB')
        image_np = np.array(image)[:, :, ::-1]  # RGB -> BGR for OpenCV/ultralytics

        model = st.session_state['model']
        results = model.predict(source=image_np, verbose=False)
        if not results:
            st.warning("No results returned by model.")
            return
        res = results[0]
        boxes = res.boxes

        # Draw boxes
        annotated = draw_boxes_on_image(image_np, boxes)
        annotated_rgb = annotated[:, :, ::-1]
        out_img = Image.fromarray(annotated_rgb)

        # Compute severity
        image_severity = 0
        detections = []
        for box in boxes:
            cls = int(box.cls[0])
            name = VOC_TO_YOLO_CLASSES[cls]
            conf = float(box.conf[0])
            severity_score = SEVERITY_MAP.get(name, 0)
            image_severity = max(image_severity, severity_score)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({"class": name, "confidence": round(conf, 3), "bbox": f"{x1},{y1},{x2},{y2}", "severity": SEVERITY_LABEL[severity_score]})

        st.image(out_img, caption="Annotated image", use_container_width=True)
        st.markdown(f"**Overall severity:** {SEVERITY_LABEL[image_severity]}")

        if detections:
            st.subheader("Detections")
            st.table(detections)

        buf = io.BytesIO()
        out_img.save(buf, format='PNG')
        buf.seek(0)
        st.download_button("Download annotated image", data=buf, file_name="annotated.png", mime="image/png")


if __name__ == '__main__':
    main()
