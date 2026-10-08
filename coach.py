"""Coach opcional por Groq; siempre devuelve un consejo local si la API falla."""
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from services import fixed_coach

LOG = logging.getLogger('ballit.coach')


def error_detail(exc, key):
    """Show the provider's reason without printing credentials or an unbounded HTML page."""
    text = exc.read(8192).decode('utf-8', errors='replace')
    try:
        body = json.loads(text)
        error = body.get('error', body)
        if isinstance(error, dict):
            text = str(error.get('code', error.get('type', ''))) + ': ' + str(error.get('message', error))
    except (ValueError, AttributeError):
        pass
    if key:
        text = text.replace(key, '[CLAVE OCULTA]')
    text = re.sub(r'gsk_[A-Za-z0-9_-]+', '[CLAVE OCULTA]', text)
    return ' '.join(text.split())[:600]


def advise(result, cache_dir: Path):
    fallback = fixed_coach(result['shots'], result['config'].get('goal'))
    key = os.getenv('GROQ_API_KEY', '').strip()
    if not key:
        return fallback
    data = {k: result[k] for k in ('config', 'rule', 'shots', 'summary', 'quality')}
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache = cache_dir / (digest + '.coach.json')
    if cache.is_file():
        return json.loads(cache.read_text(encoding='utf-8'))
    instruction = ('Eres un entrenador prudente de tiro libre. Responde exclusivamente JSON con las claves '
                   'resumen, lo_hiciste_bien (lista), puedes_mejorar (lista de objetos con titulo, que_pasa, '
                   'por_que_importa, intenta_esto, prioridad), sobre_tu_objetivo, progreso y pedir_regrabar. '
                   'No infieras si el balón entró. La regla fue calibrada con pocos videos; no diagnostiques.')
    payload = {'model': os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b'), 'temperature': 0.2,
               'response_format': {'type': 'json_object'},
               'messages': [{'role': 'system', 'content': instruction},
                            {'role': 'user', 'content': json.dumps(data, ensure_ascii=False)}]}
    request = Request('https://api.groq.com/openai/v1/chat/completions',
                      data=json.dumps(payload).encode(),
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                               'Accept': 'application/json', 'User-Agent': 'BallItMVP/0.1'})
    try:
        with urlopen(request, timeout=20) as response:
            body = json.load(response)
        answer = json.loads(body['choices'][0]['message']['content'])
        if not isinstance(answer, dict) or not all(k in answer for k in fallback):
            raise ValueError('Respuesta incompleta')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(answer, ensure_ascii=False), encoding='utf-8')
        return answer
    except HTTPError as exc:
        LOG.warning('Groq rechazó el coach: HTTP %s; %s', exc.code, error_detail(exc, key))
        return fallback
    except Exception as exc:
        LOG.warning('Coach externo no disponible: %s', exc)
        return fallback
