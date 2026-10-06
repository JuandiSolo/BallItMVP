"""Contrato JSON y lógica compartida entre Streamlit y la API de BallIt."""
import json
import math
from pathlib import Path

import cv2
import numpy as np

import analyze_elbow as A
import experiment as E
from src.angle_calculator import LANDMARK_IDS


def clean(value):
    """Convierte escalares de NumPy y NaN en JSON estricto."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def dimensions(video_path: Path, cache_path: Path):
    meta = cache_path.with_suffix('.meta.json')
    if meta.exists():
        data = json.loads(meta.read_text(encoding='utf-8'))
        if data.get('width', 0) > 0 and data.get('height', 0) > 0:
            return int(data['width']), int(data['height'])
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w <= 0 or h <= 0:
        raise ValueError('No se pudo leer el tamaño del video')
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({'width': w, 'height': h}), encoding='utf-8')
    return w, h


def normalized_frames(df, arm, width, height):
    _, sm = E._smoothed(df)
    names = list(LANDMARK_IDS)
    frames = []
    for row in df.itertuples(index=False):
        i = int(row.frame)
        if i not in sm.index:
            frames.append({'i': i, 't': float(row.timestamp_s), 'ok': False, 'p': None, 'angle': None})
            continue
        data = sm.loc[i]
        points = []
        for name in names:
            x, y = data.get(f'{name}_x'), data.get(f'{name}_y')
            points.append([max(0., min(1., float(x) / width)), max(0., min(1., float(y) / height))]
                          if np.isfinite(x) and np.isfinite(y) else None)
        angle = data.get(f'{arm}_shoulder_angle')
        frames.append({'i': i, 't': float(row.timestamp_s), 'ok': True, 'p': points,
                       'angle': float(angle) if np.isfinite(angle) else None})
    return names, frames


def fixed_coach(shots, goal=None):
    if not shots:
        return {'resumen': 'No se detectó un tiro para evaluar.', 'lo_hiciste_bien': [],
                'puedes_mejorar': [], 'sobre_tu_objetivo': None, 'progreso': None, 'pedir_regrabar': True}
    good = sum(s['verdict'] == 'bueno' for s in shots)
    items = []
    if good < len(shots):
        items.append({'titulo': 'Alineación del brazo', 'que_pasa': 'La apertura medida se acerca o supera el límite.',
                      'por_que_importa': 'Puede cambiar la posición del brazo antes del lanzamiento.',
                      'intenta_esto': 'Graba varios tiros de frente y observa el codo debajo del balón.', 'prioridad': 1})
    return {'resumen': f'{good} de {len(shots)} tiros quedaron del lado bueno de la regla provisional.',
            'lo_hiciste_bien': ['El brazo se elevó durante el tiro.'] if good else [],
            'puedes_mejorar': items, 'sobre_tu_objetivo': goal or None, 'progreso': None,
            'pedir_regrabar': False}


def shot_advice(shot, feature_label, symbol):
    out = []
    if shot['verdict'] == 'malo':
        out.append(f"**Revisa el codo:** {feature_label}: {shot['value']:.2f}{symbol}. La medición queda del lado "
                   'de los ejemplos etiquetados con codo abierto. Revisa su alineación debajo del balón.')
    elif shot['verdict'] == 'dudoso':
        out.append('**Cerca del límite:** el codo está algo abierto. Vigila que quede debajo del balón.')
    elif shot['verdict'] == 'bueno':
        out.append('**Buen codo:** queda alineado con el hombro antes del tiro.')
    if not shot['has_pause']:
        out.append('**Sin pausa detectada:** la heurística no encontró una pausa clara; esto no indica por sí solo '
                   'un problema técnico.')
    return out


def change_direction(before_mean, after_mean, good_is_lower, min_change):
    improvement = (before_mean - after_mean) if good_is_lower else (after_mean - before_mean)
    return 'mejoro' if improvement >= min_change else 'empeoro' if improvement <= -min_change else 'igual'


def result_contract(analysis_id, user_id, created_at, config, info, rule, df, width, height, video_name):
    names, frames = normalized_frames(df, config['arm'], width, height)
    shots = []
    for s in info['shots']:
        f = s['features']
        n = int(s['n'])
        shots.append({'n': n, 'value': s['value'], 'score': s['score'], 'verdict': s['verdict'],
                      'has_pause': s['has_pause'], 'pause_s': s['pause_s'],
                      'frames': {k: f[f'{k}_frame'] for k in ('start', 'set', 'release', 'prep', 'peak')},
                      'times_s': {'set': s['set_s'], 'release': s['release_s']},
                      'frame_url': f'/files/{user_id}/{analysis_id}/shot{n}.jpg' if s.get('tile') is not None else None,
                      'clip_url': None})
    warnings = []
    if config['camera'] != 'frente':
        warnings.append('La regla fue calibrada de frente; esta perspectiva puede cambiar la medición.')
    if info['detection_rate'] < 80:
        warnings.append('Se detectó a la persona en menos del 80% de los frames.')
    feature = rule['feature']
    unit = 'grados' if feature.startswith('abduction_') else 'anchos de hombro'
    result = {'analysis_id': analysis_id, 'created_at': created_at, 'config': config,
              'video': {'url': f'/files/{user_id}/{analysis_id}/{video_name}', 'width': width, 'height': height,
                        'fps': info['fps'], 'n_frames': info['n_frames'], 'duration_s': info['duration_s'],
                        'available': True},
              'quality': {'detection_rate': info['detection_rate'], 'note': info['note'] or None, 'warnings': warnings},
              'rule': {k: rule[k] for k in ('feature', 'direction', 't', 'band', 'good_mean', 'good_sd',
                                           'bad_mean', 'bad_sd')},
              'shots': shots, 'summary': info['summary'], 'landmark_names': names, 'frames': frames,
              'coach': fixed_coach(shots, config.get('goal'))}
    result['rule'].update({'angle_name': f"{config['arm']}_shoulder_angle" if feature.startswith('abduction_') else None,
                           'unit': unit})
    return clean(result)


def compare(before, after, min_change):
    a, b = before['summary'], after['summary']
    if not a['n'] or not b['n']:
        raise ValueError('Ambos análisis necesitan al menos un tiro')
    delta = {'mean': round(b['mean'] - a['mean'], 3), 'score': b['score'] - a['score'],
             'pct_bueno': round(b['pct_bueno'] - a['pct_bueno'], 2)}
    direction = before['rule']['direction']
    verdict = change_direction(a['mean'], b['mean'], direction == '<', min_change)
    short = lambda x: {k: x[k] for k in ('mean', 'score', 'pct_bueno', 'n')}
    return clean({'before': short(a), 'after': short(b), 'delta': delta, 'verdict': verdict,
                  'min_change': min_change, 'warnings': ['Compara con la misma cámara y encuadre'],
                  'before_frame_url': before['shots'][0]['frame_url'],
                  'after_frame_url': after['shots'][0]['frame_url']})
