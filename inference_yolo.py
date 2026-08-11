import argparse
import pathlib
import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on sample images.")
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=pathlib.Path("runs/train/rdd_yolov8n/weights/best.pt"),
        help="Path to a trained YOLO model.",
    )
    parser.add_argument(
        "--images",
        type=pathlib.Path,
        default=pathlib.Path("testImages"),
        help="Directory with images to run inference on.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("outputs"),
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    return parser.parse_args()


def draw_boxes(image_path: pathlib.Path, results, output_path: pathlib.Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not open {image_path}")

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = f"{cls}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    if not args.images.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images}")

    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)

    image_files = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {args.images}")

    print(f"Running inference on {len(image_files)} images...")
    for image_path in image_files[:10]:
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            save=False,
            verbose=False,
        )
        output_path = args.output / image_path.name
        draw_boxes(image_path, results, output_path)
        print(f"Saved {output_path}")

    print("Inference complete")


if __name__ == "__main__":
    main()
