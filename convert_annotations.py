import argparse
import pathlib
import xml.etree.ElementTree as ET

VOC_TO_YOLO_CLASSES = ["Block crack", "D00", "D10", "D20", "D40", "Repair"]


def convert_xml_to_yolo(xml_path: pathlib.Path, yolo_path: pathlib.Path) -> None:
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

    yolo_path.parent.mkdir(parents=True, exist_ok=True)
    yolo_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VOC XML annotations to YOLO txt format.")
    parser.add_argument(
        "--annotations-dir",
        type=pathlib.Path,
        default=pathlib.Path("RDD2022_China_Drone/annotations/xmls"),
        help="Directory containing Pascal VOC XML annotation files.",
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
        default=pathlib.Path("RDD2022_China_Drone/yolo_labels"),
        help="Output directory for YOLO label txt files.",
    )

    args = parser.parse_args()
    annotations = sorted(args.annotations_dir.glob("*.xml"))
    if not annotations:
        raise FileNotFoundError(f"No XML files found in {args.annotations_dir}")

    print(f"Converting {len(annotations)} annotations...")
    for xml_path in annotations:
        filename = xml_path.stem
        yolo_path = args.output_dir / f"{filename}.txt"
        convert_xml_to_yolo(xml_path, yolo_path)

    print(f"Converted {len(annotations)} annotations to YOLO format in {args.output_dir}")


if __name__ == "__main__":
    main()
