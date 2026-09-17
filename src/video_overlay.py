"""
video_overlay.py
-------------------
Dibuja los valores de los angulos como texto directamente sobre el video,
al lado de cada articulacion, para que el usuario "vea" el analisis.
"""

import cv2

# Color en BGR (OpenCV usa BGR, no RGB)
TEXT_COLOR = (0, 255, 255)      # amarillo
BG_COLOR = (0, 0, 0)            # fondo negro detras del texto, para que se lea
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Que punto usamos como "ancla" visual para mostrar el texto de cada angulo
ANCHOR_FOR_ANGLE = {
    "left_elbow_angle": "left_elbow",
    "right_elbow_angle": "right_elbow",
    "left_knee_angle": "left_knee",
    "right_knee_angle": "right_knee",
    "left_hip_angle": "left_hip",
    "right_hip_angle": "right_hip",
    "left_shoulder_angle": "left_shoulder",
    "right_shoulder_angle": "right_shoulder",
}


def draw_angle_labels(frame, points: dict, angles: dict):
    """
    Escribe el valor numerico del angulo al lado de cada articulacion.
    `points` son las coordenadas en pixeles (de landmarks_to_xy).
    `angles` son los angulos calculados (de compute_all_angles).
    """
    for angle_name, anchor_name in ANCHOR_FOR_ANGLE.items():
        value = angles.get(angle_name)
        if value is None or anchor_name not in points:
            continue

        x, y = int(points[anchor_name][0]), int(points[anchor_name][1])
        label = f"{int(value)} deg"

        # Fondo negro simple detras del texto para que se lea sobre cualquier color
        (text_w, text_h), _ = cv2.getTextSize(label, FONT, 0.5, 2)
        cv2.rectangle(frame, (x, y - text_h - 4), (x + text_w + 4, y), BG_COLOR, -1)
        cv2.putText(frame, label, (x + 2, y - 2), FONT, 0.5, TEXT_COLOR, 2, cv2.LINE_AA)

    return frame


def draw_frame_counter(frame, frame_idx: int, total_frames: int, timestamp_s: float):
    """Pequeno contador arriba a la izquierda, util para ubicar momentos clave despues."""
    label = f"Frame {frame_idx}/{total_frames}  t={timestamp_s:.2f}s"
    cv2.putText(frame, label, (10, 25), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame
