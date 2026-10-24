.\.venv\Scripts\Activate.ps1

python scripts\visualize_labels.py --images images\train\video3 --labels labels\train\video3 --output results\train\video3

python scripts\add_prefix.py -f labels\train\video3 -p v3