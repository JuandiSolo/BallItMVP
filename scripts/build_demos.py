#!/usr/bin/env python3
"""
build_demos.py
--------------
Script offline para construir los 3 videos demo de BallIt.

Uso:
  python scripts/build_demos.py --scan demo_sources        # 1) Analiza videos crudos, muestra tabla
  python scripts/build_demos.py --build                    # 2) Construye demos/ según manifest.json
  python scripts/build_demos.py --build --no-groq          #    (opcional) coach genérico local
"""
import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Asegurar que el path incluye la raíz del back
BACK_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACK_ROOT))

import analyze_elbow as A
import coach
import experiment as E
import media
import services

# Config
LOG = logging.getLogger('build_demos')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

SUFFIXES = {'.mp4', '.mov', '.m4v', '.avi', '.mkv'}
CREATED_AT = '2025-01-01T00:00:00+00:00'  # constante para JSONs estables entre builds
DEMOS_DIR = BACK_ROOT / 'demos'
MANIFEST_PATH = DEMOS_DIR / 'manifest.json'
SOURCES_DIR = BACK_ROOT / 'demo_sources'
RULE_PATH = BACK_ROOT / 'elbow_out' / 'rule.json'

# Textos por defecto según veredicto global
DEFAULT_TEXTS = {
    'bueno': {
        'title': 'Tiro con buena mecánica',
        'description': 'El codo queda alineado bajo el balón. Así se ve un resultado verde.'
    },
    'dudoso': {
        'title': 'Tiro cerca del límite',
        'description': 'El codo está algo abierto. Mira qué detecta BallIt y cómo lo explica.'
    },
    'malo': {
        'title': 'Tiro con el codo abierto',
        'description': 'El codo se separa del cuerpo. Mira cómo lo marca BallIt y qué te sugiere.'
    }
}


def load_manifest():
    if not MANIFEST_PATH.exists():
        return {'version': 1, 'demos': []}
    return json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))


def save_manifest(manifest):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


def load_rule():
    rule = A.load_rule(RULE_PATH.parent)
    if rule is None:
        raise RuntimeError(f'Falta {RULE_PATH} (regla obligatoria). Corre analyze_elbow.py --videos ... primero.')
    if rule.get('source', '').startswith('regla de ejemplo'):
        raise RuntimeError('La regla es la de ejemplo; no se pueden construir demos sin calibrar. '
                           'Corre analyze_elbow.py --videos ... con tus videos etiquetados.')
    return rule


def global_verdict(shots):
    """Veredicto global: el más frecuente; en empate, el más severo (malo > dudoso > bueno)."""
    if not shots:
        return 'malo'
    counts = {'bueno': 0, 'dudoso': 0, 'malo': 0}
    for s in shots:
        counts[s['verdict']] = counts.get(s['verdict'], 0) + 1
    max_count = max(counts.values())
    candidates = [k for k, v in counts.items() if v == max_count]
    severity = {'malo': 3, 'dudoso': 2, 'bueno': 1}
    return max(candidates, key=lambda k: severity[k])


def scan_video(video_path, arm='right'):
    """Analiza un video y devuelve info para la tabla de --scan."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Path(tmpdir) / 'cache.csv'
        info = A.analyze_clip(video_path, cache, arm, load_rule(),
                              frac=float(load_rule().get('frac', 0.90)),
                              prep_level=float(load_rule().get('prep_level', 0.60)))
    return info


def transcode_video(source, output, trim_start=None, trim_end=None):
    """Transcodifica a H.264 720p max, yuv420p, +faststart, sin audio."""
    cmd = ['ffmpeg', '-loglevel', 'error', '-y']
    if trim_start is not None:
        cmd += ['-ss', str(trim_start)]
    cmd += ['-i', str(source)]
    if trim_end is not None and trim_start is not None:
        cmd += ['-t', str(trim_end - trim_start)]
    cmd += [
        '-vf', "scale=-2:'min(720,ih)'",
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-an',
        str(output)
    ]
    subprocess.run(cmd, check=True, timeout=180, capture_output=True)


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'No se pudo abrir {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or count <= 0:
        raise RuntimeError(f'Video inválido: {video_path}')
    return {'fps': fps, 'n_frames': int(count), 'width': w, 'height': h, 'duration_s': count / fps}


def validate_durations(video_path, annotated_path):
    """Valida que video.mp4 y annotated.mp4 duren lo mismo (< 0.1s diff)."""
    v1 = get_video_info(video_path)
    v2 = get_video_info(annotated_path)
    diff = abs(v1['duration_s'] - v2['duration_s'])
    if diff >= 0.1:
        raise RuntimeError(f'Duraciones no coinciden: video={v1["duration_s"]:.2f}s, annotated={v2["duration_s"]:.2f}s (diff={diff:.2f}s)')
    return v1


def render_annotated(source, output, sm, arm, fps):
    """Renderiza TODOS los frames con esqueleto y ángulo (versión completa de render_clip)."""
    # Extraer la lógica de render_clip para renderizar rango completo
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir el video para crear el anotado')
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = output.with_name(output.stem + '_raw.mp4')
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    try:
        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ok, frame = cap.read()
            if not ok:
                break
            if i in sm.index:
                frame = media.draw(frame, sm.loc[i], arm, i)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    # Recodificar a H.264
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(raw),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', str(output)],
                   check=True, timeout=180, capture_output=True)
    raw.unlink(missing_ok=True)


def make_thumbnail(video_path, output, frame_idx, target_w=480):
    """Extrae un frame y lo guarda como JPEG 480px ancho, calidad ~80."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'No se pudo leer frame {frame_idx} de {video_path}')
    h, w = frame.shape[:2]
    new_w = min(target_w, w)
    new_h = int(h * new_w / w)
    thumb = cv2.resize(frame, (new_w, new_h))
    cv2.imwrite(str(output), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])


def build_demo(slug, demo_cfg, rule, arm, no_groq):
    """Construye un demo individual."""
    LOG.info(f'=== Construyendo demo: {slug} ===')

    demo_dir = DEMOS_DIR / slug
    demo_dir.mkdir(parents=True, exist_ok=True)

    source = BACK_ROOT / demo_cfg['source']
    if not source.exists():
        raise FileNotFoundError(f'Video fuente no encontrado: {source}')

    # 1. Transcodificar
    video_path = demo_dir / 'video.mp4'
    LOG.info('Transcodificando video...')
    transcode_video(source, video_path, demo_cfg.get('trim_start_s'), demo_cfg.get('trim_end_s'))
    vinfo = get_video_info(video_path)
    LOG.info(f'  {vinfo["width"]}x{vinfo["height"]} @ {vinfo["fps"]:.1f}fps, {vinfo["duration_s"]:.1f}s')

    # 2. Cargar regla y analizar
    cache_dir = Path(tempfile.mkdtemp(prefix=f'build_{slug}_'))
    cache = cache_dir / 'cache.csv'

    try:
        LOG.info('Analizando video...')
        info = A.analyze_clip(video_path, cache, arm, rule,
                              frac=float(rule.get('frac', 0.90)),
                              prep_level=float(rule.get('prep_level', 0.60)))

        if not info['shots']:
            raise RuntimeError(f'{slug}: no se detectó ningún tiro; reemplaza el video')

        info['summary'] = A.summarize_shots(info['shots'])

        # 3. Construir result_contract
        df = pd.read_csv(cache)
        w, h = services.dimensions(video_path, cache)

        aid = str(uuid.uuid5(uuid.NAMESPACE_URL, f'ballit-demo:{slug}'))
        config = {
            'arm': arm,
            'camera': 'frente',
            'focus': 'completo',
            'goal': None,
            'trim_start_s': None,
            'trim_end_s': None
        }

        result = services.result_contract(aid, 'demo', CREATED_AT, config, info, rule, df, w, h, 'video.mp4')

        # 4. Generar tiros (jpg + mp4)
        LOG.info('Generando tiros...')
        for shot, measured in zip(result['shots'], info['shots']):
            n = shot['n']
            if measured['tile'] is not None:
                cv2.imwrite(str(demo_dir / f'shot{n}.jpg'), measured['tile'])

        _, sm = E._smoothed(df)
        for shot in result['shots']:
            n = shot['n']
            try:
                media.render_clip(video_path, demo_dir / f'shot{n}.mp4', sm, arm, info['fps'],
                                  shot['frames']['start'], shot['frames']['release'])
            except Exception as exc:
                LOG.warning(f'No se pudo generar clip tiro {n}: {exc}')
                result['quality']['warnings'].append(f'No se pudo generar el clip del tiro {n}.')

        # 5. Coach con Groq
        LOG.info('Obteniendo coach...')
        if not no_groq:
            from dotenv import load_dotenv
            load_dotenv(BACK_ROOT / '.env')
            key = os.getenv('GROQ_API_KEY', '').strip()
            if not key:
                raise RuntimeError('GROQ_API_KEY no está en .env. Usa --no-groq o configura la clave.')
            result['coach'] = coach.advise(result, cache_dir)

            # Detectar fallback silencioso
            fallback = services.fixed_coach(result['shots'], config.get('goal'))
            if result['coach'] == fallback:
                raise RuntimeError('Groq no respondió (fallback genérico). Revisa la clave/red o usa --no-groq.')
            LOG.info(f'  Coach: Groq ({os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")})')
        else:
            result['coach'] = services.fixed_coach(result['shots'], config.get('goal'))
            LOG.info('  Coach: genérico local (--no-groq)')

        # 6. Video anotado
        LOG.info('Renderizando video anotado...')
        annotated_path = demo_dir / 'annotated.mp4'
        render_annotated(video_path, annotated_path, sm, arm, info['fps'])
        validate_durations(video_path, annotated_path)
        LOG.info('  Duraciones validadas OK')

        # 7. Miniatura (frame 'set' del primer tiro)
        first_shot = info['shots'][0]
        thumb_path = demo_dir / 'thumb.jpg'
        make_thumbnail(video_path, thumb_path, first_shot['features']['set_frame'])

        # 8. Reescribir URLs y añadir campo demo
        base = f'/demos/{slug}/files'
        result['video']['url'] = f'{base}/video.mp4'
        result['video']['available'] = True
        for shot in result['shots']:
            n = shot['n']
            shot['frame_url'] = f'{base}/shot{n}.jpg'
            shot['clip_url'] = f'{base}/shot{n}.mp4'

        verdict = global_verdict(result['shots'])
        score = result['summary']['score']

        demo_meta = {
            'slug': slug,
            'title': demo_cfg.get('title') or DEFAULT_TEXTS[verdict]['title'],
            'description': demo_cfg.get('description') or DEFAULT_TEXTS[verdict]['description'],
            'thumb_url': f'{base}/thumb.jpg',
            'video_url': f'{base}/video.mp4',
            'annotated_url': f'{base}/annotated.mp4',
            'duration_s': round(vinfo['duration_s'], 1),
            'n_shots': result['summary']['n'],
            'score': score,
            'verdict': verdict
        }
        result['demo'] = demo_meta

        # 9. Guardar result.json
        result_path = demo_dir / 'result.json'
        result_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')

        # 10. Actualizar manifest con textos por defecto si eran null
        if demo_cfg.get('title') is None:
            demo_cfg['title'] = demo_meta['title']
        if demo_cfg.get('description') is None:
            demo_cfg['description'] = demo_meta['description']

        # Tamaño
        total_mb = sum(f.stat().st_size for f in demo_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        LOG.info(f'  ✓ {slug}: verdict={verdict}, score={score}, {total_mb:.1f} MB')

        return demo_meta, total_mb

    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def cmd_scan(args):
    """Modo --scan: analiza videos en demo_sources y muestra tabla."""
    if not SOURCES_DIR.exists():
        LOG.error(f'No existe {SOURCES_DIR}. Crea la carpeta y pon ahí tus videos.')
        sys.exit(1)

    videos = [p for p in sorted(SOURCES_DIR.rglob('*')) if p.suffix.lower() in SUFFIXES]
    if not videos:
        LOG.error(f'No hay videos en {SOURCES_DIR} (extensiones: {", ".join(SUFFIXES)})')
        sys.exit(1)

    rule = load_rule()
    arm = args.arm

    print(f'\n{"Archivo":<30} {"Dur":>5} {"Res":>10} {"Det%":>5} {"Tiros":>5} {"Score":>5} {"Veredicto":>8} {"Pausa"}')
    print('-' * 95)

    for vid in videos:
        try:
            info = scan_video(vid, arm)
            det_rate = info['detection_rate']
            n_shots = len(info['shots'])

            if n_shots == 0:
                code = 'NO_PERSON' if det_rate < 15 else 'NO_SHOT'
                note = info['note'] or ('No se ve al jugador' if code == 'NO_PERSON' else 'No se detectó tiro')
                print(f'{vid.name:<30} {"-":>5} {"-":>10} {det_rate:>5.1f} {"0":>5} {"-":>5} {code:>8} -')
                print(f'  → {note}')
            else:
                for shot in info['shots']:
                    print(f'{vid.name:<30} {info["duration_s"]:>5.1f} {info.get("n_frames",0)//info.get("fps",30) if info.get("fps") else "-":>10} {det_rate:>5.1f} {n_shots:>5} {shot["score"]:>5} {shot["verdict"]:>8} {"sí" if shot["has_pause"] else "no"}')
        except Exception as e:
            print(f'{vid.name:<30} ERROR: {e}')

    print('\nRecomendación: elige idealmente 3 videos con veredictos distintos (verde/amarillo/rojo) y 5–15 s de duración.')
    print('Luego edita demos/manifest.json con los slugs y fuentes elegidos, y ejecuta --build.')


def cmd_build(args):
    """Modo --build: construye demos según manifest.json."""
    manifest = load_manifest()
    if not manifest.get('demos'):
        LOG.error('manifest.json está vacío. Ejecuta --scan primero y rellena el manifest.')
        sys.exit(1)

    rule = load_rule()
    arm = manifest.get('arm', 'right')  # por defecto right

    total_mb = 0
    for demo_cfg in manifest['demos']:
        slug = demo_cfg['slug']
        demo_meta, mb = build_demo(slug, demo_cfg, rule, demo_cfg.get('arm', arm), args.no_groq)
        total_mb += mb

    # Guardar manifest actualizado (con titles/descriptions rellenos)
    save_manifest(manifest)

    if total_mb > 25:
        LOG.warning(f'demos/ pesa {total_mb:.1f} MB (> 25 MB). Considera videos más cortos o menor bitrate.')
    else:
        LOG.info(f'demos/ total: {total_mb:.1f} MB')

    LOG.info('¡Build completado!')


def main():
    ap = argparse.ArgumentParser(description='Construye los demos de BallIt')
    ap.add_argument('--arm', default='right', choices=['left', 'right'], help='Brazo por defecto para --scan')
    sub = ap.add_subparsers(dest='cmd', required=True)

    scan_p = sub.add_parser('scan', help='Escanea videos en demo_sources y muestra tabla')
    scan_p.add_argument('source', nargs='?', default='demo_sources', help='Carpeta con videos crudos')

    build_p = sub.add_parser('build', help='Construye demos/ según manifest.json')
    build_p.add_argument('--no-groq', action='store_true', help='Usa coach genérico local (no llama a Groq)')

    args = ap.parse_args()

    if args.cmd == 'scan':
        # Permitir source personalizado
        global SOURCES_DIR
        SOURCES_DIR = BACK_ROOT / args.source
        cmd_scan(args)
    elif args.cmd == 'build':
        cmd_build(args)


if __name__ == '__main__':
    main()