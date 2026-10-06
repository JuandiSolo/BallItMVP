"""Crea clips H.264 con el esqueleto y el ángulo dibujados."""
import subprocess
from pathlib import Path

import cv2
import numpy as np

import experiment as E
from src.angle_calculator import LANDMARK_IDS


def draw(frame, data, arm, index):
    points = {}
    for name in LANDMARK_IDS:
        x, y = data.get(f'{name}_x'), data.get(f'{name}_y')
        if np.isfinite(x) and np.isfinite(y):
            points[name] = (int(x), int(y))
    for side in ('left', 'right'):
        for a, b in E.CHAINS:
            p, q = points.get(f'{side}_{a}'), points.get(f'{side}_{b}')
            if p and q:
                cv2.line(frame, p, q, (0, 220, 0) if side == arm else (190, 190, 190),
                         4 if side == arm else 2, cv2.LINE_AA)
    shoulder = points.get(f'{arm}_shoulder')
    angle = data.get(f'{arm}_shoulder_angle')
    if shoulder and np.isfinite(angle):
        cv2.putText(frame, f'Apertura del codo: {float(angle):.1f} deg',
                    (max(4, shoulder[0] - 50), max(25, shoulder[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Frame {index}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def render_clip(source: Path, output: Path, sm, arm, fps, start, release):
    pad_before, pad_after = int(.5 * fps), int(1.0 * fps)
    first, last = max(0, start - pad_before), release + pad_after
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir el video para crear el clip')
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = output.with_name(output.stem + '_raw.mp4')
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    count = 0
    try:
        for i in range(first, last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if i in sm.index:
                frame = draw(frame, sm.loc[i], arm, i)
            writer.write(frame)
            count += 1
    finally:
        writer.release()
        cap.release()
    if not count:
        raw.unlink(missing_ok=True)
        raise RuntimeError('El recorte no contiene frames')
    try:
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(raw), '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(output)],
                       check=True, timeout=120, capture_output=True)
    finally:
        raw.unlink(missing_ok=True)
