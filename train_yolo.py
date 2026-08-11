import argparse
import pathlib
import subprocess
import sys
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on RDD2022 dataset.")
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        default=pathlib.Path("data.yaml"),
        help="YOLO dataset config file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base YOLO model to use for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training.",
    )
    parser.add_argument(
        "--clean-after",
        action="store_true",
        help="Run clean_builds.py to remove older runs after training (keeps most recent).",
    )
    parser.add_argument(
        "--project",
        type=pathlib.Path,
        default=pathlib.Path("runs/train"),
        help="Project directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="rdd_yolov8n",
        help="Experiment name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")

    print(f"Training YOLO model {args.model} with data config {args.data}")
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    print("Training complete")

    if args.clean_after:
        try:
            print("Cleaning older runs...")
            subprocess.run(
                [sys.executable, "clean_builds.py", "--project", str(args.project), "--keep", "1", "--yes"],
                check=True,
            )
            print("Cleaning finished.")
        except Exception as e:
            print("Warning: cleaning failed:", e)


if __name__ == "__main__":
    main()
