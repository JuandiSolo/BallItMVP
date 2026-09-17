"""
pose_extractor.py
-------------------
Envuelve MediaPipe Pose. Su unico trabajo es: dado un frame de video,
devolver los landmarks (puntos del cuerpo) detectados, o None si no
detecto a nadie en ese frame.
"""

import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class PoseExtractor:
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        """
        model_complexity: 0 (lite, mas rapido), 1 (default), 2 (mas preciso, mas lento)
        Para un video ya grabado (no en vivo), model_complexity=2 es buena idea
        porque no necesitamos tiempo real.
        """
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr):
        """
        Recibe un frame en formato BGR (el que da OpenCV por defecto).
        Devuelve el resultado de MediaPipe (result.pose_landmarks puede ser None
        si no detecto a nadie en el frame).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # optimizacion recomendada por MediaPipe
        result = self.pose.process(frame_rgb)
        return result

    def draw_landmarks(self, frame_bgr, result):
        """Dibuja el esqueleto sobre el frame (in-place)."""
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame_bgr

    def close(self):
        self.pose.close()
