"""
angle_calculator.py
--------------------
Calcula angulos entre articulaciones a partir de 3 puntos (landmarks).

La idea geometrica es simple: un angulo articular (ej. el codo) se define
por 3 puntos -> hombro (a), codo (b), muneca (c). El codo (b) es el vertice.
Calculamos el angulo entre los vectores b->a y b->c usando la formula del
coseno (producto punto).
"""

import numpy as np

# Indices de los landmarks de MediaPipe Pose que nos interesan para basquet.
# Referencia completa: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARK_IDS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

# Cada angulo articular se define como (punto_a, vertice, punto_c)
ANGLE_DEFINITIONS = {
    "left_elbow_angle":  ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle":   ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle":  ("right_hip", "right_knee", "right_ankle"),
    "left_hip_angle":    ("left_shoulder", "left_hip", "left_knee"),
    "right_hip_angle":   ("right_shoulder", "right_hip", "right_knee"),
    "left_shoulder_angle":  ("left_hip", "left_shoulder", "left_elbow"),
    "right_shoulder_angle": ("right_hip", "right_shoulder", "right_elbow"),
}


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calcula el angulo en el vertice `b`, formado por los puntos a-b-c.
    Los puntos son arrays [x, y] (usamos solo 2D, que es lo mas estable
    para video de una sola camara).

    Devuelve el angulo en grados (0-180).
    """
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    # Clip para evitar errores de redondeo que manden el valor fuera de [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return float(np.degrees(angle_rad))


def landmarks_to_xy(landmarks, frame_width: int, frame_height: int) -> dict:
    """
    Convierte los landmarks normalizados de MediaPipe (0.0 a 1.0) a
    coordenadas de pixeles reales, para poder dibujarlos y medirlos.
    """
    points = {}
    for name, idx in LANDMARK_IDS.items():
        lm = landmarks[idx]
        points[name] = np.array([lm.x * frame_width, lm.y * frame_height])
    return points


def compute_all_angles(points: dict) -> dict:
    """
    Dado un diccionario de puntos {nombre: [x, y]}, calcula todos los
    angulos definidos en ANGLE_DEFINITIONS.
    """
    angles = {}
    for angle_name, (p_a, p_b, p_c) in ANGLE_DEFINITIONS.items():
        if p_a in points and p_b in points and p_c in points:
            angles[angle_name] = round(
                calculate_angle(points[p_a], points[p_b], points[p_c]), 1
            )
        else:
            angles[angle_name] = None
    return angles
