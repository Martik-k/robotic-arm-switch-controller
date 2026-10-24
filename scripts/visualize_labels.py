from argparse import ArgumentParser
from pathlib import Path

import cv2


CLASS_NAMES = {0: "main_on", 1: "main_off", 2: "main_on", 3: "main_off"}
COLORS = {0: (0, 200, 0), 1: (0, 0, 255), 2: (0, 200, 0), 3: (0, 0, 255)}


def draw_labels(image_path: Path, label_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    if label_path.exists():
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"{label_path}:{line_number}: expected 5 fields")

            class_id, x_center, y_center, box_width, box_height = map(float, fields)
            class_id = int(class_id)
            if class_id not in CLASS_NAMES:
                raise ValueError(f"{label_path}:{line_number}: unknown class {class_id}")

            x1 = round((x_center - box_width / 2) * width)
            y1 = round((y_center - box_height / 2) * height)
            x2 = round((x_center + box_width / 2) * width)
            y2 = round((y_center + box_height / 2) * height)
            color = COLORS[class_id]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                CLASS_NAMES[class_id],
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write image: {output_path}")


def find_label_path(image_path: Path, labels_dir: Path) -> Path:
    exact_path = labels_dir / f"{image_path.stem}.txt"
    if exact_path.exists():
        return exact_path

    if "_" in image_path.stem:
        bfs_export_name = image_path.stem.split("_", 1)[1] + ".txt"
        fallback_path = labels_dir / bfs_export_name
        if fallback_path.exists():
            return fallback_path

    return exact_path


def main() -> None:
    parser = ArgumentParser(description="Draw YOLO labels on matching images.")
    parser.add_argument("--images", type=Path, default=Path("images/train/video1"))
    parser.add_argument("--labels", type=Path, default=Path("labels/train/video1"))
    parser.add_argument("--output", type=Path, default=Path("results/train/video1"))
    args = parser.parse_args()

    image_paths = sorted(
        path for path in args.images.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for image_path in image_paths:
        label_path = find_label_path(image_path, args.labels)
        draw_labels(image_path, label_path, args.output / image_path.name)

    print(f"Saved {len(image_paths)} annotated images to {args.output}")


if __name__ == "__main__":
    main()
