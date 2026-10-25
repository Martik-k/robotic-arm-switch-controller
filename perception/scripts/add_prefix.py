"""Utility script to prefix filenames (e.g. v1_, v2_) to avoid naming collisions across videos."""

import argparse
import os


def add_prefix_to_files(folder_path: str, prefix: str) -> None:
    # Ensure target path exists
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist!")
        return

    # Ensure delimiter suffix (e.g., 'v1_')
    clean_prefix = prefix.strip()
    if not clean_prefix.endswith("_"):
        clean_prefix += "_"

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not files:
        print(f"No files found to rename in directory '{folder_path}'.")
        return

    renamed_count = 0
    skipped_count = 0

    print(f"Starting file renaming in directory: {folder_path}")
    print(f"Prefix: '{clean_prefix}'\n")

    for filename in files:
        # Avoid duplicate prefixing
        if filename.startswith(clean_prefix):
            skipped_count += 1
            continue

        old_path = os.path.join(folder_path, filename)
        new_filename = f"{clean_prefix}{filename}"
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        renamed_count += 1

    print(
        f"Done! Renamed: {renamed_count} files. Skipped (already had prefix): {skipped_count}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add a prefix (e.g., v1_, v2_) to all files within a directory."
    )
    parser.add_argument(
        "--folder", "-f", type=str, help="Path to the target directory containing files."
    )
    parser.add_argument(
        "--prefix", "-p", type=str, help="Prefix string to prepend (e.g., v1, v2)."
    )

    args = parser.parse_args()

    # Prompt interactively if arguments are omitted from CLI
    folder = args.folder or input(
        "Enter target folder path (e.g. images/train/video1): "
    ).strip()
    prefix = args.prefix or input("Enter prefix (e.g. v1): ").strip()

    add_prefix_to_files(folder, prefix)