import os
os.environ['BALLIT_CORS_ORIGINS'] = 'https://ballit.vercel.app,http://localhost:8080,http://localhost:5500'
from fastapi.testclient import TestClient
import api

c = TestClient(api.app)

def preflight(origin):
    r = c.options('/demos', headers={
        'Origin': origin,
        'Access-Control-Request-Method': 'GET',
    })
    return r.status_code, r.headers.get('access-control-allow-origin'), r.headers.get('access-control-allow-methods')

def simple_get(origin):
    r = c.get('/demos', headers={'Origin': origin})
    return r.status_code, r.headers.get('access-control-allow-origin')

origins = [
    'https://ballit.vercel.app',
    'http://localhost:8080',
    'http://localhost:5500',
    'https://ballit-git-feature-x.vercel.app',  # preview pattern
    'https://evil.example.com',                  # must be rejected
]

print('== PREFLIGHT OPTIONS /demos ==')
for o in origins:
    print(o, '->', preflight(o))

print('\n== GET /demos ==')
for o in origins:
    print(o, '->', simple_get(o))

print('\n== GET /demos/{slug} preflight ==')
r = c.options('/demos/tiro-1', headers={'Origin': 'https://ballit.vercel.app', 'Access-Control-Request-Method': 'GET'})
print('status', r.status_code, 'allow-origin', r.headers.get('access-control-allow-origin'))

print('\n== RANGE on demo video ==')
r = c.get('/demos/tiro-1/files/video.mp4', headers={'Range': 'bytes=0-99', 'Origin': 'https://ballit.vercel.app'})
print('status', r.status_code, 'content-range', r.headers.get('content-range'), 'accept-ranges', r.headers.get('accept-ranges'), 'allow-origin', r.headers.get('access-control-allow-origin'))
