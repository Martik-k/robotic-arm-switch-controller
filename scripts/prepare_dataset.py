"""Prepare flattened YOLO dataset in data/ with 0-indexed classes.

Maps legacy class IDs:
    Class 3 (main_off) -> Class 0 (main_off)
    Class 2 (main_on)  -> Class 1 (main_on)
"""

import shutil
from pathlib import Path

# Mapping from legacy CVAT/BFS class IDs to switches.yaml 0-indexed classes
CLASS_MAPPING = {
    3: 0,  # main_off (vertical, ~6 o'clock)
    2: 1,  # main_on  (horizontal, ~3 o'clock)
    1: 0,  # fallback if 0/1 was already used
    0: 1,  # fallback if legacy 0=on, 1=off was used
}


def prepare_split(
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    split_name: str,
) -> tuple[int, int]:
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    img_count = 0
    box_count = 0

    # Search all subdirectories (video1, video2, video3, etc.)
    image_files = sorted(
        [
            p
            for p in src_images_dir.rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )

    for img_path in image_files:
        # Find matching label file
        # The label file is in the corresponding subdirectory of labels
        rel_path = img_path.relative_to(src_images_dir)
        label_candidate = src_labels_dir / rel_path.with_suffix(".txt")

        # Destination paths (flattened by filename into train/ or val/)
        dst_img_path = dst_images_dir / img_path.name
        dst_lbl_path = dst_labels_dir / f"{img_path.stem}.txt"

        # Copy image if not exists or different size
        if (
            not dst_img_path.exists()
            or dst_img_path.stat().st_size != img_path.stat().st_size
        ):
            shutil.copy2(img_path, dst_img_path)

        # Process labels and remap classes
        new_lines = []
        if label_candidate.exists():
            for line in label_candidate.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                new_cls = CLASS_MAPPING.get(old_cls, old_cls)

                coords = parts[1:]
                new_lines.append(f"{new_cls} " + " ".join(coords))
                box_count += 1

        dst_lbl_path.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8"
        )
        img_count += 1

    print(
        f"[{split_name}] Processed {img_count} images and {box_count} switch boxes -> {dst_images_dir}"
    )
    return img_count, box_count


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    # Train split: video1 and video2
    prepare_split(
        src_images_dir=root / "images" / "train",
        src_labels_dir=root / "labels" / "train",
        dst_images_dir=root / "data" / "images" / "train",
        dst_labels_dir=root / "data" / "labels" / "train",
        split_name="train",
    )

    # Val split: video3
    prepare_split(
        src_images_dir=root / "images" / "val",
        src_labels_dir=root / "labels" / "val",
        dst_images_dir=root / "data" / "images" / "val",
        dst_labels_dir=root / "data" / "labels" / "val",
        split_name="val",
    )

    print("\nDataset preparation complete! 'data/' is ready for YOLO training.")


if __name__ == "__main__":
    main()
