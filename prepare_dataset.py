import argparse
import random
import shutil
import pathlib
import xml.etree.ElementTree as ET

VOC_TO_YOLO_CLASSES = ["Block crack", "D00", "D10", "D20", "D40", "Repair"]


def convert_annotation(xml_path: pathlib.Path, output_path: pathlib.Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in VOC_TO_YOLO_CLASSES:
            raise ValueError(f"Unknown class '{name}' in {xml_path}")

        class_id = VOC_TO_YOLO_CLASSES.index(name)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(annotations_dir: pathlib.Path, images_dir: pathlib.Path, output_dir: pathlib.Path, val_split: float, test_split: float, seed: int, extensions=None) -> None:
    if extensions is None:
        extensions = ['.jpg', '.png', '.jpeg']

    annotations = sorted(annotations_dir.glob("*.xml"))
    if not annotations:
        raise FileNotFoundError(f"No XML files found in {annotations_dir}")

    # Map annotation stems to existing image files with supported extensions.
    image_names = []
    missing_images = []
    for xml in annotations:
        found = False
        for ext in extensions:
            candidate = images_dir / (xml.stem + ext)
            if candidate.exists():
                image_names.append(candidate.name)
                found = True
                break
        if not found:
            missing_images.append(xml.stem)

    if not image_names:
        raise FileNotFoundError(f"No dataset images found in {images_dir} for extensions {extensions}")
    if missing_images:
        print(f"Warning: {len(missing_images)} annotations have no matching image and will be skipped (examples: {missing_images[:5]})")
    random.seed(seed)
    random.shuffle(image_names)

    num_images = len(image_names)
    num_test = int(num_images * test_split)
    num_val = int(num_images * val_split)
    num_train = num_images - num_val - num_test

    splits = {
        "train": image_names[:num_train],
        "val": image_names[num_train:num_train + num_val],
        "test": image_names[num_train + num_val:],
    }

    for split_name, image_list in splits.items():
        images_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for image_name in image_list:
            source_image = images_dir / image_name
            if not source_image.exists():
                print(f"Skipping missing image: {source_image}")
                continue

            target_image = images_out / image_name
            shutil.copy2(source_image, target_image)

            xml_path = annotations_dir / (source_image.stem + ".xml")
            label_path = labels_out / (source_image.stem + ".txt")
            try:
                convert_annotation(xml_path, label_path)
            except Exception as e:
                print(f"Warning: failed to convert annotation {xml_path}: {e}")

    config_path = output_dir / "data.yaml"
    names_block = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(VOC_TO_YOLO_CLASSES))
    config_path.write_text(
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"\nnc: {len(VOC_TO_YOLO_CLASSES)}\n"
        f"names:\n{names_block}\n",
        encoding="utf-8",
    )
    print(f"Prepared dataset at {output_dir} with {num_train} train, {num_val} val, {num_test} test images")
    print(f"Created YAML config: {config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RDD2022 dataset for YOLO training.")
    parser.add_argument(
        "--annotations-dir",
        type=pathlib.Path,
        default=pathlib.Path("RDD2022_China_Drone/annotations/xmls"),
        help="Pascal VOC XML annotations directory.",
    )
    parser.add_argument(
        "--images-dir",
        type=pathlib.Path,
        default=pathlib.Path("RDD2022_China_Drone/images"),
        help="Directory containing dataset images.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("RDD2022_China_Drone/yolo_dataset"),
        help="Destination directory for prepared dataset.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data used for validation.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data used for test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(args.annotations_dir, args.images_dir, args.output_dir, args.val_split, args.test_split, args.seed)


if __name__ == "__main__":
    main()
