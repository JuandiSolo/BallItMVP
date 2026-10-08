"""API local de BallIt. Ejecutar: uvicorn api:app --host 0.0.0.0 --port 8000"""
import sys

if sys.version_info < (3, 10):
    raise RuntimeError("BallIt API requiere Python 3.10–3.12. Activa un entorno con Python 3.11 y vuelve a iniciar Uvicorn.")

import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import cv2
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load the backend's own file, regardless of the directory used to start Uvicorn.
# Variables already exported in the terminal take precedence.
load_dotenv(Path(__file__).resolve().parent / '.env', override=False)

import analyze_elbow as A
import coach
import experiment as E
import media
import services

LOG = logging.getLogger('ballit')
ROOT = Path(os.getenv('BALLIT_DATA_DIR', Path(__file__).parent)).resolve()
VIDEO_ROOT, RESULT_ROOT, CACHE_ROOT = (ROOT / name for name in ('videos_prueba', 'resultados', 'cache'))
MAX_BYTES = int(os.getenv('BALLIT_MAX_BYTES', 200 * 1024 * 1024))
MAX_SECONDS = float(os.getenv('BALLIT_MAX_SECONDS', 0))
if not math.isfinite(MAX_SECONDS) or MAX_SECONDS < 0:
    raise ValueError('BALLIT_MAX_SECONDS debe ser 0 (sin límite) o un número positivo')
SUFFIXES = {'.mp4', '.mov', '.m4v', '.avi', '.mkv'}
CORS_ORIGINS = [x.strip() for x in os.getenv('BALLIT_CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173').split(',') if x.strip()]
app = FastAPI(title='BallIt API', version='0.1.0')
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=False,
                   allow_methods=['GET', 'POST', 'DELETE'], allow_headers=['X-User-Id', 'Content-Type'])


@app.exception_handler(HTTPException)
async def http_error(_request, exc):
    detail = exc.detail if isinstance(exc.detail, dict) else {'code': 'BAD_REQUEST', 'message': str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=detail)


@app.exception_handler(RequestValidationError)
async def validation_error(_request, _exc):
    return JSONResponse(status_code=422, content={'code': 'BAD_INPUT', 'message': 'Revisa los campos enviados'})
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ballit-analysis')
lock = threading.Lock()
jobs = {}
rule = A.load_rule(Path(__file__).parent / 'elbow_out') or dict(A.DEFAULT_RULE)
if rule.get('source'):
    LOG.warning('Falta elbow_out/rule.json: se usa una regla de ejemplo en otra escala')


def error(status, code, message):
    raise HTTPException(status_code=status, detail={'code': code, 'message': message})


def user_id(x_user_id: str = Header(..., alias='X-User-Id')) -> str:
    try:
        return str(uuid.UUID(x_user_id))
    except (ValueError, AttributeError):
        error(400, 'INVALID_USER_ID', 'X-User-Id debe ser un UUID válido')


def analysis_id(value):
    try:
        return str(uuid.UUID(value))
    except ValueError:
        error(400, 'BAD_ID', 'El identificador del análisis no es válido')


def result_path(user, aid):
    return RESULT_ROOT / user / (aid + '.json')


def write_result(user, aid, result):
    p = result_path(user, aid)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(p)


def read_result(user, aid):
    p = result_path(user, aid)
    if not p.is_file():
        error(404, 'NOT_FOUND', 'No se encontró este análisis')
    result = json.loads(p.read_text(encoding='utf-8'))
    video = result['video']
    video['available'] = (VIDEO_ROOT / user / aid / Path(video['url']).name).is_file()
    return result


def set_job(aid, **fields):
    with lock:
        jobs[aid].update(fields)


def duration(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        error(400, 'BAD_FORMAT', 'No pude abrir el video')
    fps, count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0 or count <= 0:
        error(400, 'BAD_FORMAT', 'No pude leer la duración del video')
    return count / fps


def process(user, aid, source, config, trim_start, trim_end):
    work_video = source
    folder = source.parent
    try:
        set_job(aid, status='processing', progress=0., step='Preparando video')
        if trim_start is not None or trim_end is not None:
            work_video = folder / 'segment.mp4'
            cmd = ['ffmpeg', '-loglevel', 'error', '-y']
            if trim_start is not None:
                cmd += ['-ss', str(trim_start)]
            cmd += ['-i', str(source)]
            if trim_end is not None:
                cmd += ['-t', str(trim_end - (trim_start or 0))]
            cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', str(work_video)]
            subprocess.run(cmd, check=True, timeout=120, capture_output=True)
        hasher = hashlib.sha256()
        with work_video.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        cache = CACHE_ROOT / (digest + '.csv')
        set_job(aid, step='Detectando movimiento')
        def progress(value):
            set_job(aid, progress=round(.05 + .75 * value, 3))
        info = A.analyze_clip(work_video, cache, config['arm'], rule,
                              frac=float(rule.get('frac', .90)), prep_level=float(rule.get('prep_level', .60)),
                              progress=progress)
        if not info['shots']:
            code = 'NO_PERSON' if info['detection_rate'] < 15 else 'NO_SHOT'
            message = info['note'] or ('No se ve al jugador: cuerpo completo y buena luz' if code == 'NO_PERSON'
                                        else 'No pude ver un tiro: que se vea cuando sube el balón')
            set_job(aid, status='error', step='Error', error={'code': code, 'message': message})
            return
        info['summary'] = A.summarize_shots(info['shots'])
        df = pd.read_csv(cache)
        width, height = services.dimensions(work_video, cache)
        result = services.result_contract(aid, user, datetime.now(timezone.utc).isoformat(), config,
                                          info, rule, df, width, height, work_video.name)
        result['coach'] = coach.advise(result, CACHE_ROOT)
        for shot, measured in zip(result['shots'], info['shots']):
            if measured['tile'] is not None:
                cv2.imwrite(str(folder / f"shot{shot['n']}.jpg"), measured['tile'])
        write_result(user, aid, result)
        set_job(aid, status='done', progress=1., step='Listo', result=result, media_pending=True)
        # Los clips se publican a medida que terminan; un error no invalida el análisis.
        _, sm = E._smoothed(df)
        for shot in result['shots']:
            n = shot['n']
            try:
                media.render_clip(work_video, folder / f'shot{n}.mp4', sm, config['arm'], info['fps'],
                                  shot['frames']['start'], shot['frames']['release'])
                shot['clip_url'] = f'/files/{user}/{aid}/shot{n}.mp4'
                write_result(user, aid, result)
                set_job(aid, result=result)
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                LOG.warning('No se pudo generar el clip %s/%s: %s', aid, n, exc)
                result['quality']['warnings'].append(f'No se pudo generar el clip del tiro {n}.')
                write_result(user, aid, result)
    except Exception:
        LOG.exception('Fallo de análisis %s', aid)
        set_job(aid, status='error', step='Error', error={'code': 'ANALYSIS_FAILED',
                'message': 'No pude analizar este video, intenta de nuevo'})
    finally:
        set_job(aid, media_pending=False)


@app.post('/analyze', status_code=202)
async def analyze(video: UploadFile = File(...), arm: Literal['right', 'left'] = Form(...),
                  camera: Literal['frente', 'lateral', 'diagonal'] = Form('frente'),
                  focus: Literal['brazo', 'postura', 'completo'] = Form('completo'),
                  goal: str = Form(''), trim_start_s: float | None = Form(None),
                  trim_end_s: float | None = Form(None), user: str = Depends(user_id)):
    suffix = Path(video.filename or '').suffix.lower()
    if suffix not in SUFFIXES:
        error(400, 'BAD_FORMAT', 'Sube un video mp4, mov, m4v, avi o mkv')
    if trim_start_s is not None and (not math.isfinite(trim_start_s) or trim_start_s < 0):
        error(400, 'BAD_TRIM', 'El inicio del recorte debe ser finito y mayor o igual a cero')
    if trim_end_s is not None and (not math.isfinite(trim_end_s) or trim_end_s <= (trim_start_s or 0)):
        error(400, 'BAD_TRIM', 'El final del recorte debe ser posterior al inicio')
    if len(goal) > 500:
        error(400, 'BAD_GOAL', 'El objetivo es demasiado largo')
    aid = str(uuid.uuid4())
    folder = VIDEO_ROOT / user / aid
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / ('video' + suffix)
    total = 0
    try:
        with source.open('wb') as out:
            while chunk := await video.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_BYTES:
                    error(413, 'VIDEO_TOO_LARGE', 'El video es muy pesado')
                out.write(chunk)
        length = duration(source)
        if trim_end_s is not None and trim_end_s > length:
            error(400, 'BAD_TRIM', 'El final del recorte supera la duración del video')
        if trim_start_s is not None and trim_start_s >= length:
            error(400, 'BAD_TRIM', 'El recorte comienza después del video')
        segment = (trim_end_s or length) - (trim_start_s or 0)
        if MAX_SECONDS > 0 and segment > MAX_SECONDS:
            error(400, 'VIDEO_TOO_LONG', f'El segmento supera el límite configurado de {MAX_SECONDS:g} s')
    except Exception:
        source.unlink(missing_ok=True)
        folder.rmdir()
        raise
    config = {'arm': arm, 'camera': camera, 'focus': focus, 'goal': goal or None,
              'trim_start_s': trim_start_s, 'trim_end_s': trim_end_s}
    with lock:
        jobs[aid] = {'user': user, 'status': 'processing', 'progress': 0., 'step': 'En cola'}
    executor.submit(process, user, aid, source, config, trim_start_s, trim_end_s)
    return {'analysis_id': aid, 'status': 'processing'}


@app.get('/analyze/{id}')
def get_analysis(id: str, user: str = Depends(user_id)):
    aid = analysis_id(id)
    with lock:
        job = jobs.get(aid)
        state = dict(job) if job and job['user'] == user else None
    if state:
        state.pop('user', None)
        if state['status'] == 'done':
            state['result'] = read_result(user, aid)
        return state
    if result_path(user, aid).exists():
        return {'status': 'done', 'progress': 1., 'step': 'Listo', 'result': read_result(user, aid)}
    error(404, 'NOT_FOUND', 'No se encontró este análisis')


@app.get('/history')
def history(user: str = Depends(user_id)):
    folder = RESULT_ROOT / user
    items = []
    for path in folder.glob('*.json') if folder.exists() else []:
        item = read_result(user, path.stem)
        summary = item['summary']
        items.append({'analysis_id': item['analysis_id'], 'created_at': item['created_at'],
                      **{k: summary[k] for k in ('n', 'mean', 'score', 'pct_bueno')},
                      'video': {'available': item['video']['available']}})
    return sorted(items, key=lambda x: x['created_at'], reverse=True)


@app.get('/history/{id}')
def history_item(id: str, user: str = Depends(user_id)):
    return read_result(user, analysis_id(id))


class Comparison(BaseModel):
    before_id: str
    after_id: str
    min_change: float | None = None


@app.post('/compare')
def compare(payload: Comparison, user: str = Depends(user_id)):
    before = read_result(user, analysis_id(payload.before_id))
    after = read_result(user, analysis_id(payload.after_id))
    if before['rule']['feature'] != after['rule']['feature'] or before['config']['arm'] != after['config']['arm']:
        error(400, 'INCOMPATIBLE_ANALYSES', 'Los análisis deben usar la misma métrica y mano')
    minimum = payload.min_change if payload.min_change is not None else before['rule']['band']
    if minimum < 0:
        error(400, 'BAD_MIN_CHANGE', 'El cambio mínimo debe ser positivo')
    return services.compare(before, after, minimum)


@app.get('/files/{user_path}/{id}/{filename}')
def files(user_path: str, id: str, filename: str, user: str = Depends(user_id)):
    if user_path != user:
        error(404, 'NOT_FOUND', 'No se encontró el archivo')
    aid = analysis_id(id)
    if not re.fullmatch(r'(video\.(mp4|mov|m4v|avi|mkv)|segment\.mp4|shot[1-9][0-9]*\.(jpg|mp4))', filename):
        error(404, 'NOT_FOUND', 'No se encontró el archivo')
    path = VIDEO_ROOT / user / aid / filename
    if not path.is_file() or not result_path(user, aid).is_file():
        error(404, 'NOT_FOUND', 'No se encontró el archivo')
    return FileResponse(path, media_type=mimetypes.guess_type(filename)[0] or 'application/octet-stream')


@app.delete('/history/{id}')
def delete_history(id: str, user: str = Depends(user_id)):
    aid = analysis_id(id)
    with lock:
        job = jobs.get(aid)
        if job and job['user'] == user and (job['status'] == 'processing' or job.get('media_pending')):
            error(409, 'IN_PROGRESS', 'Espera a que termine el análisis y los clips')
    p = result_path(user, aid)
    if not p.is_file():
        error(404, 'NOT_FOUND', 'No se encontró este análisis')
    p.unlink()
    with lock:
        jobs.pop(aid, None)
    return {'deleted': True}
