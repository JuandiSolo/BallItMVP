"""Limpia videos subidos sin eliminar resultados ni historiales.

Ejemplo cron mensual: 0 3 1 * * cd /ruta/BallItMVP && python3 limpiar_videos.py
"""
import argparse
import os
import shutil
import time
from pathlib import Path


def clean(root: Path, days: int = 30, max_bytes: int = 0, now=None):
    now = time.time() if now is None else now
    items = [p for p in root.glob('*/*') if p.is_dir()]
    removed = []
    for item in items:
        latest = max((p.stat().st_mtime for p in item.rglob('*') if p.is_file()), default=item.stat().st_mtime)
        if now - latest > days * 86400:
            shutil.rmtree(item)
            removed.append(str(item))
    if max_bytes:
        remaining = [p for p in root.glob('*/*') if p.is_dir()]
        size = lambda p: sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
        total = sum(size(p) for p in remaining)
        for item in sorted(remaining, key=lambda p: p.stat().st_mtime):
            if total <= max_bytes:
                break
            total -= size(item)
            shutil.rmtree(item)
            removed.append(str(item))
    return removed


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Elimina videos antiguos, conserva resultados')
    ap.add_argument('--days', type=int, default=30)
    ap.add_argument('--max-gb', type=float, default=0)
    ap.add_argument('--root', type=Path, default=Path(os.getenv('BALLIT_DATA_DIR', Path(__file__).parent)) / 'videos_prueba')
    args = ap.parse_args()
    if args.days < 0 or args.max_gb < 0:
        ap.error('Los límites deben ser positivos')
    for path in clean(args.root, args.days, int(args.max_gb * 1024**3)):
        print(path)
