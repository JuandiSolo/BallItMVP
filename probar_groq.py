"""Run from the backend: python probar_groq.py. Never prints the API key."""
import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def main():
    path = Path(__file__).resolve().parent / '.env'
    exported = bool(os.getenv('GROQ_API_KEY'))
    load_dotenv(path, override=False)
    key = os.getenv('GROQ_API_KEY', '').strip()
    print('Archivo .env:', 'encontrado' if path.is_file() else 'NO encontrado')
    print('Origen de la clave:', 'terminal (tiene prioridad sobre .env)' if exported else '.env')
    print('Clave configurada:', 'sí' if key else 'NO')
    if not key:
        return 1
    model = os.getenv('GROQ_MODEL', 'openai/gpt-oss-20b')
    print('Modelo:', model)
    payload = {'model': model, 'temperature': 0.2, 'response_format': {'type': 'json_object'},
               'messages': [{'role': 'user', 'content': 'Responde únicamente el objeto JSON {"ok":true}.'}]}
    request = Request('https://api.groq.com/openai/v1/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
                               'Accept': 'application/json', 'User-Agent': 'BallItMVP/0.1'})
    try:
        with urlopen(request, timeout=20) as response:
            body = json.load(response)
        content = body['choices'][0]['message']['content']
        json.loads(content)
        print('OK: Groq respondió y devolvió JSON. La conexión funciona.')
        return 0
    except HTTPError as exc:
        detail = exc.read(8192).decode('utf-8', errors='replace').replace(key, '[CLAVE OCULTA]')
        detail = re.sub(r'gsk_[A-Za-z0-9_-]+', '[CLAVE OCULTA]', detail)
        print('ERROR HTTP:', exc.code)
        print('Motivo:', ' '.join(detail.split())[:1000])
        if exc.code == 403:
            print('Si dice model_permission_blocked, habilita el modelo en Settings > Limits del proyecto y la organización.')
            print('Si muestra HTML o un bloqueo de seguridad, revisa la red, VPN o proxy desde el que se envía la petición.')
        return 1
    except (URLError, TimeoutError) as exc:
        print('No se pudo conectar a Groq:', str(exc).replace(key, '[CLAVE OCULTA]'))
        return 1
    except (KeyError, IndexError, ValueError):
        print('Groq respondió, pero la respuesta no tiene el JSON esperado.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
