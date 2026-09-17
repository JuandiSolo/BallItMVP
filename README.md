# Basketball Pose MVP

MVP que analiza un video de un movimiento de básquet (o cualquier movimiento
humano), extrae el esqueleto con **MediaPipe**, calcula los ángulos de las
articulaciones clave, dibuja el análisis sobre el video, y genera un resumen
de texto listo para pasarle a una IA (como Claude) y obtener feedback técnico.

## Qué está pasando, en resumen

```
video de entrada (mp4)
        │
        ▼
┌───────────────────┐   MediaPipe detecta 33 puntos del cuerpo
│  pose_extractor.py │   (hombros, codos, muñecas, caderas, rodillas...)
└───────────────────┘   en cada frame del video
        │
        ▼
┌────────────────────┐  Con geometría simple (vectores + coseno),
│ angle_calculator.py│  convierte esos puntos en ángulos:
└────────────────────┘  "el codo derecho está a 145°"
        │
        ▼
┌────────────────────┐  Dibuja el esqueleto + el número de cada
│  video_overlay.py  │  ángulo directamente sobre el video
└────────────────────┘
        │
        ▼
┌────────────────────┐  Guarda TODOS los ángulos, frame por frame,
│  data_exporter.py  │  en JSON y CSV (la "base de datos" cruda)
└────────────────────┘
        │
        ▼
┌──────────────────────┐  Resume esos cientos de números en una
│ summary_generator.py │  tabla corta (mínimo/máximo/rango por
└──────────────────────┘  articulación) y arma el prompt final
        │
        ▼
   prompt_para_ia.txt  ──►  (lo pegas en Claude, o lo mandas
                             por API) para obtener el análisis
                             técnico en lenguaje natural
```

**Por qué se separa así:** MediaPipe te da coordenadas de puntos (x, y), no
"ángulos" ni "feedback". Ese es un trabajo de geometría que hacemos nosotros
(`angle_calculator.py`). Y una IA de lenguaje como Claude no puede "ver" el
video directamente de forma útil — por eso le pasamos un **resumen numérico
ya procesado**, no el video ni los cientos de frames crudos.

## Árbol de archivos

```
basketball-pose-mvp/
├── main.py                    # Punto de entrada (CLI). Orquesta todo el pipeline.
├── requirements.txt           # Dependencias de Python
├── README.md                  # Este archivo
│
├── src/
│   ├── pose_extractor.py      # Envuelve MediaPipe: da los puntos del cuerpo por frame
│   ├── angle_calculator.py    # Convierte puntos -> ángulos (matemática pura)
│   ├── video_overlay.py       # Dibuja los ángulos como texto sobre el video
│   ├── data_exporter.py       # Guarda los datos en JSON y CSV
│   └── summary_generator.py   # Resume los datos y arma el prompt para la IA
│
├── sample_data/                # Aquí pones el video que quieras analizar
│   └── test_synthetic.mp4      # Video de prueba SIN persona real (solo para
│                                # verificar que el código no truena; no
│                                # detectará ningún esqueleto)
│
└── output/                     # Se genera automáticamente al correr main.py
    ├── analisis_video.mp4      # Video ORIGINAL + esqueleto + ángulos dibujados
    ├── angulos_por_frame.json  # Datos crudos: un registro por cada frame
    ├── angulos_por_frame.csv   # Los mismos datos, en formato tabla (Excel/Sheets)
    ├── resumen.md               # Tabla resumen: min/máx/rango de cada articulación
    └── prompt_para_ia.txt       # Texto listo para copiar y pegar a Claude (o API)
```

## Instalación

Necesitas Python 3.9 o superior.

```bash
cd basketball-pose-mvp
pip install -r requirements.txt
```

> **Nota importante sobre la versión de MediaPipe:** el `requirements.txt`
> fija `mediapipe==0.10.14` a propósito. Las versiones más nuevas de
> MediaPipe (0.10.30 en adelante) **eliminaron** la API sencilla
> `mediapipe.solutions.pose` que este proyecto usa, y la reemplazaron por
> una API más compleja (`mediapipe.tasks`) que requiere descargar un
> archivo de modelo `.task` por separado. Si más adelante quieres migrar a
> la API nueva (recomendable a largo plazo, ya que la vieja eventualmente
> desaparecerá del todo), avísame y te ayudo a adaptar `pose_extractor.py`.

## Cómo probar el prototipo

### 1. Consigue un video de prueba

Graba un video corto (5-15 segundos) de alguien haciendo el movimiento que
quieres analizar (un tiro, un salto, un dribleo). Recomendaciones para
mejores resultados:

- **Cámara fija** (no la muevas mientras grabas)
- El **cuerpo completo** debe verse en cuadro todo el tiempo
- **Buena luz**, sin contraluz
- Grábalo de **perfil o en 3/4** si te interesan ángulos de brazo/pierna
  específicos (de frente algunos ángulos se distorsionan)
- Formatos soportados: `.mp4`, `.mov`, `.avi` (cualquier formato que lea OpenCV)

Guarda el video en `sample_data/`, por ejemplo `sample_data/mi_tiro.mp4`.

### 2. Corre el análisis

```bash
python main.py --input sample_data/mi_tiro.mp4 --output output --label "tiro libre"
```

Parámetros disponibles:

| Parámetro | Qué hace | Default |
|---|---|---|
| `--input` | Ruta al video de entrada (obligatorio) | — |
| `--output` | Carpeta donde se guardan los resultados | `output` |
| `--label` | Descripción del movimiento (se usa en el resumen y el prompt) | `"tiro de basquet"` |
| `--complexity` | Precisión del modelo: `0` (rápido), `1` (default), `2` (más preciso, más lento) | `1` |

Ejemplo con más precisión (recomendado si el video no es en vivo, ya que no
necesitas tiempo real):

```bash
python main.py --input sample_data/mi_tiro.mp4 --complexity 2 --label "salto vertical"
```

### 3. Revisa los resultados

En la consola vas a ver algo así:

```
Video: 1080x1920 @ 30.0fps, 145 frames
Procesados 145 frames en 12.3s (11.8 fps de procesamiento)
Persona detectada en 94.5% de los frames
```

- **"Persona detectada en X%"**: si este número es bajo (menos de 70-80%),
  probablemente el encuadre, la luz o la distancia de la cámara no son
  ideales — vuelve a grabar ajustando eso.

Abre `output/analisis_video.mp4` para **ver el esqueleto y los ángulos
dibujados sobre tu video**. Abre `output/resumen.md` para ver la tabla
resumen. Abre `output/prompt_para_ia.txt`, copia el contenido, y pégalo en
una conversación con Claude (o mándalo por la API) para obtener el análisis
técnico y las recomendaciones.

### 4. (Opcional) Prueba rápida sin video propio

Si solo quieres confirmar que el código corre sin errores antes de grabar tu
propio video, ya incluí `sample_data/test_synthetic.mp4`. **Ojo:** ese video
NO tiene una persona real (es una figura sintética), así que la detección va
a dar 0% — sirve solo para confirmar que el pipeline no truena, no para ver
resultados reales.

```bash
python main.py --input sample_data/test_synthetic.mp4 --label "prueba"
```

## Ángulos que se calculan actualmente

| Ángulo | Puntos usados (hombro-codo-muñeca, etc.) |
|---|---|
| Codo izquierdo / derecho | hombro → codo → muñeca |
| Rodilla izquierda / derecha | cadera → rodilla → tobillo |
| Cadera izquierda / derecha | hombro → cadera → rodilla |
| Hombro izquierdo / derecho | cadera → hombro → codo |

Si quieres agregar más (ej. ángulo de la muñeca, inclinación del torso),
solo hay que añadir una entrada nueva en `ANGLE_DEFINITIONS` dentro de
`src/angle_calculator.py` — no hay que tocar nada más, el resto del
pipeline lo recoge automáticamente.

## Limitaciones actuales (a tener en cuenta)

- **Una sola cámara = ángulos en 2D.** MediaPipe estima una tercera
  dimensión (profundidad), pero no es tan confiable como usar 2+ cámaras.
  Para básquet esto suele ser suficiente si grabas de perfil, pero ten en
  cuenta que un movimiento que va "hacia" la cámara puede leerse distorsionado.
- **No hay validación de "ángulo ideal" todavía.** El código calcula los
  ángulos y se los pasa a la IA para que ella dé el criterio técnico; el
  código en sí no sabe qué es "bueno" o "malo" en biomecánica.
- **No hay detección automática del "momento clave"** (ej. el instante
  exacto del release del tiro). El resumen actual da el mínimo/máximo de
  todo el video. Si quieres que identifique automáticamente fases del
  movimiento (preparación, release, seguimiento), es el siguiente paso
  natural a construir.
- **No está conectado a la API de IA todavía.** El MVP genera el
  `prompt_para_ia.txt` listo para usar, pero llamar automáticamente a la
  API (para no tener que copiar/pegar) es fácil de agregar — dímelo si
  quieres que lo integre.

## Siguientes pasos sugeridos

1. Probarlo con 2-3 videos reales tuyos y ver qué tan bien detecta.
2. Si la detección falla mucho, subir `--complexity` a `2` o mejorar
   condiciones de grabación.
3. Conectar `prompt_para_ia.txt` directamente a la API de Claude para que
   el feedback salga automático (sin copiar/pegar).
4. Definir rangos "ideales" de ángulos por tipo de movimiento (esto
   normalmente se saca de estudios de biomecánica deportiva o comparando
   contra jugadores profesionales).
