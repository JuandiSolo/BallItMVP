"""
data_exporter.py
-------------------
Guarda los datos crudos (angulo por frame) en JSON y CSV.
Esta es la "base de datos" del movimiento que despues se resume
para mandarsela a la IA.
"""

import json
import csv
from pathlib import Path


def save_json(frames_data: list, output_path: str):
    """
    frames_data: lista de dicts, uno por frame, con forma:
    {
      "frame": 0,
      "timestamp_s": 0.0,
      "detected": True,
      "angles": {"left_elbow_angle": 145.2, ...}
    }
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(frames_data, f, indent=2, ensure_ascii=False)


def save_csv(frames_data: list, output_path: str):
    """Mismo contenido que save_json pero en formato tabular, util para Excel/Sheets."""
    if not frames_data:
        return

    angle_names = sorted(frames_data[0]["angles"].keys())
    fieldnames = ["frame", "timestamp_s", "detected"] + angle_names

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in frames_data:
            flat_row = {
                "frame": row["frame"],
                "timestamp_s": row["timestamp_s"],
                "detected": row["detected"],
            }
            flat_row.update(row["angles"])
            writer.writerow(flat_row)
