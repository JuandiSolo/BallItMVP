#!/usr/bin/env python3
"""
main.py
---------
Punto de entrada del MVP. Orquesta todo el pipeline:

  video de entrada
        |
        v
  [pose_extractor]   -> detecta el esqueleto en cada frame
        |
        v
  [angle_calculator] -> calcula angulos de codo, rodilla, cadera, hombro
        |
        v
  [video_overlay]    -> dibuja el esqueleto + angulos sobre el video
        |
        v
  [data_exporter]     -> guarda todo en JSON/CSV
        |
        v
  [summary_generator] -> genera un resumen en Markdown + un prompt listo
                          para mandarle a una API de IA (Claude, etc)

Uso:
    python main.py --input sample_data/mi_video.mp4 --label "tiro libre"
"""

import argparse
import time
from pathlib import Path

import cv2

from src.pose_extractor import PoseExtractor
from src.angle_calculator import landmarks_to_xy, compute_all_angles
from src.video_overlay import draw_angle_labels, draw_frame_counter
from src.data_exporter import save_json, save_csv
from src.summary_generator import build_summary, summary_to_markdown, build_ai_prompt


def process_video(input_path: str, output_dir: str, movement_label: str,
                   model_complexity: int = 1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Video de salida con el overlay dibujado
    output_video_path = output_dir / "analisis_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    extractor = PoseExtractor(model_complexity=model_complexity)

    frames_data = []
    frame_idx = 0
    detected_count = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_s = frame_idx / fps
        result = extractor.process_frame(frame)

        if result.pose_landmarks:
            detected_count += 1
            points = landmarks_to_xy(result.pose_landmarks.landmark, width, height)
            angles = compute_all_angles(points)

            # Dibujar esqueleto + angulos sobre el frame
            extractor.draw_landmarks(frame, result)
            draw_angle_labels(frame, points, angles)

            frames_data.append({
                "frame": frame_idx,
                "timestamp_s": round(timestamp_s, 3),
                "detected": True,
                "angles": angles,
            })
        else:
            # No se detecto a nadie en este frame (ej. jugador fuera de cuadro)
            frames_data.append({
                "frame": frame_idx,
                "timestamp_s": round(timestamp_s, 3),
                "detected": False,
                "angles": {k: None for k in [
                    "left_elbow_angle", "right_elbow_angle",
                    "left_knee_angle", "right_knee_angle",
                    "left_hip_angle", "right_hip_angle",
                    "left_shoulder_angle", "right_shoulder_angle",
                ]},
            })

        draw_frame_counter(frame, frame_idx, total_frames, timestamp_s)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    extractor.close()

    elapsed = time.time() - t0
    detection_rate = 100 * detected_count / max(frame_idx, 1)
    print(f"Procesados {frame_idx} frames en {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} fps de procesamiento)")
    print(f"Persona detectada en {detection_rate:.1f}% de los frames")

    # Guardar datos crudos
    save_json(frames_data, str(output_dir / "angulos_por_frame.json"))
    save_csv(frames_data, str(output_dir / "angulos_por_frame.csv"))

    # Generar resumen + prompt para IA
    summary = build_summary(frames_data, fps)
    summary_md = summary_to_markdown(summary, movement_label=movement_label)
    ai_prompt = build_ai_prompt(summary_md, movement_label=movement_label)

    (output_dir / "resumen.md").write_text(summary_md, encoding="utf-8")
    (output_dir / "prompt_para_ia.txt").write_text(ai_prompt, encoding="utf-8")

    print("\nArchivos generados en:", output_dir.resolve())
    print(" - analisis_video.mp4       (video con overlay de esqueleto y angulos)")
    print(" - angulos_por_frame.json   (datos crudos, un registro por frame)")
    print(" - angulos_por_frame.csv    (mismos datos, formato tabla)")
    print(" - resumen.md               (resumen legible: min/max/rango por articulacion)")
    print(" - prompt_para_ia.txt       (prompt listo para mandar a la API de una IA)")

    return {
        "detection_rate": detection_rate,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVP de analisis de movimiento con MediaPipe")
    parser.add_argument("--input", required=True, help="Ruta al video de entrada (mp4, mov, etc)")
    parser.add_argument("--output", default="output", help="Carpeta donde guardar resultados")
    parser.add_argument("--label", default="tiro de basquet",
                         help="Descripcion del movimiento, ej. 'tiro libre', 'salto'")
    parser.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2],
                         help="Complejidad del modelo MediaPipe: 0=rapido, 1=default, 2=preciso")
    args = parser.parse_args()

    process_video(args.input, args.output, args.label, args.complexity)
