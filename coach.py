"""Coach opcional por Groq; siempre devuelve un consejo local si la API falla."""
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.request import Request, urlopen

from services import fixed_coach

LOG = logging.getLogger('ballit.coach')


def advise(result, cache_dir: Path):
    fallback = fixed_coach(result['shots'], result['config'].get('goal'))
    key = os.getenv('GROQ_API_KEY')
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
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=5) as response:
            body = json.load(response)
        answer = json.loads(body['choices'][0]['message']['content'])
        if not isinstance(answer, dict) or not all(k in answer for k in fallback):
            raise ValueError('Respuesta incompleta')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(answer, ensure_ascii=False), encoding='utf-8')
        return answer
    except Exception as exc:
        LOG.warning('Coach externo no disponible: %s', exc)
        return fallback
