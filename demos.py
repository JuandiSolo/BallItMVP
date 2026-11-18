"""Router de demos públicos para BallIt. No requiere autenticación."""
import json
import re
from pathlib import Path
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix='', tags=['demos'])

DEMOS_DIR = Path(__file__).parent / 'demos'
MANIFEST_PATH = DEMOS_DIR / 'manifest.json'

# Caché en memoria: se carga una vez al importar
_MANIFEST = None
_RESULTS = {}  # slug -> result.json parseado


def _load_all():
    """Carga manifest y todos los result.json una sola vez."""
    global _MANIFEST, _RESULTS
    if _MANIFEST is not None:
        return

    if not MANIFEST_PATH.exists():
        _MANIFEST = {'version': 1, 'demos': []}
        return

    _MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    _RESULTS = {}

    for demo_cfg in _MANIFEST.get('demos', []):
        slug = demo_cfg['slug']
        result_path = DEMOS_DIR / slug / 'result.json'
        if result_path.exists():
            _RESULTS[slug] = json.loads(result_path.read_text(encoding='utf-8'))


def _error(status, code, message):
    raise HTTPException(status_code=status, detail={'code': code, 'message': message})


@router.get('')
def list_demos():
    """Lista de demos (orden = manifest.json)."""
    _load_all()
    items = []
    for demo_cfg in _MANIFEST.get('demos', []):
        slug = demo_cfg['slug']
        result = _RESULTS.get(slug)
        if not result:
            continue
        shots = result.get('shots', [])
        verdict = 'malo'
        if shots:
            counts = {'bueno': 0, 'dudoso': 0, 'malo': 0}
            for s in shots:
                counts[s['verdict']] = counts.get(s['verdict'], 0) + 1
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            severity = {'malo': 3, 'dudoso': 2, 'bueno': 1}
            verdict = max(candidates, key=lambda k: severity[k])

        items.append({
            'slug': slug,
            'title': demo_cfg.get('title', ''),
            'description': demo_cfg.get('description', ''),
            'thumb_url': f'/demos/{slug}/files/thumb.jpg',
            'video_url': f'/demos/{slug}/files/video.mp4',
            'annotated_url': f'/demos/{slug}/files/annotated.mp4',
            'duration_s': result.get('video', {}).get('duration_s', 0),
            'n_shots': result.get('summary', {}).get('n', 0),
            'score': result.get('summary', {}).get('score', 0),
            'verdict': verdict
        })
    return JSONResponse(content=items, headers={'Cache-Control': 'public, max-age=300'})


@router.get('/{slug}')
def get_demo(slug: str):
    """Result.json completo del demo + campo demo."""
    _load_all()
    if slug not in _RESULTS:
        _error(404, 'NOT_FOUND', 'Demo no encontrado')
    result = _RESULTS[slug]
    # Añadir campo demo si no existe (compatibilidad)
    if 'demo' not in result:
        demo_cfg = next((d for d in _MANIFEST.get('demos', []) if d['slug'] == slug), {})
        result = dict(result)
        result['demo'] = {
            'slug': slug,
            'title': demo_cfg.get('title', ''),
            'description': demo_cfg.get('description', ''),
            'thumb_url': f'/demos/{slug}/files/thumb.jpg',
            'video_url': f'/demos/{slug}/files/video.mp4',
            'annotated_url': f'/demos/{slug}/files/annotated.mp4'
        }
    return JSONResponse(content=result, headers={'Cache-Control': 'public, max-age=300'})


# Lista blanca de archivos permitidos
FILE_PATTERN = re.compile(r'^(?:(?:video|annotated)\.mp4|thumb\.jpg|shot[1-9][0-9]*\.(?:jpg|mp4))$')


@router.get('/{slug}/files/{filename}')
def get_demo_file(slug: str, filename: str):
    """Sirve archivos estáticos del demo (público, sin auth)."""
    _load_all()
    if slug not in _RESULTS:
        _error(404, 'NOT_FOUND', 'Demo no encontrado')

    if not FILE_PATTERN.match(filename):
        _error(404, 'NOT_FOUND', 'Archivo no permitido')

    path = DEMOS_DIR / slug / filename
    if not path.is_file():
        _error(404, 'NOT_FOUND', 'Archivo no encontrado')

    # Headers de cache para archivos estáticos
    headers = {'Cache-Control': 'public, max-age=86400'}
    return FileResponse(path, headers=headers)