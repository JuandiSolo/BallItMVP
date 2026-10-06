# 🏀 BallIt MVP

¡Hey! Si quieren probar lo que llevamos de BallIt, pueden correrlo en su computador y abrir la app en el navegador. Suben un video de un tiro libre grabado **de frente**, revisan la posición del codo antes de lanzar y comparan dos videos.

Por ahora es un **prototipo local**. Esta guía explica cómo instalarlo, verlo y probarlo con los videos incluidos.

## API para el front web

La API usa el mismo análisis del prototipo. Requiere **Python 3.10–3.12**, las dependencias de `requirements.txt` y `ffmpeg` con codificador `libx264` instalado en el servidor (`ffmpeg -encoders | grep libx264`). Generen primero la regla con el comando de la sección 3; `elbow_out/rule.json` queda en el proyecto. Si falta, la API usa una regla de ejemplo en otra escala y lo registra como advertencia.

En macOS, `python3` puede apuntar al Python 3.9 de Xcode. Compruébenlo con `python3 --version`. Si es 3.9 y tienen Homebrew, instalen Python 3.11 y creen un entorno nuevo desde la carpeta del proyecto:

```bash
brew install python@3.11
python3.11 -m venv .venv311
source .venv311/bin/activate
python --version  # debe decir Python 3.11.x
python -m pip install -r requirements.txt
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Al volver otro día, ejecuten `source .venv311/bin/activate` antes de iniciar la API. Usar `python -m uvicorn` con el entorno activado garantiza que Uvicorn use ese mismo Python.

```bash
python3 -m pip install -r requirements.txt
python3 analyze_elbow.py --videos videos/frente/ --arm right
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Documentación interactiva: `http://127.0.0.1:8000/docs`. Copia `example_result.json` para desarrollar las pantallas del front sin esperar a que termine un análisis.

El front debe generar y conservar un **UUID** por instalación y enviarlo en `X-User-Id` en cada petición. Ejemplo de subida:

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'X-User-Id: 550e8400-e29b-41d4-a716-446655440000' \
  -F 'video=@videos/frente/bueno/42242067-8455-4e73-8727-fce7f0ba6168.mov' \
  -F 'arm=right' -F 'camera=frente' -F 'focus=completo'
```

La respuesta `202` trae `analysis_id`. Consulta `GET /analyze/{analysis_id}` hasta recibir `status: done` y lee `result`; si llega `error`, muestra `error.message`. `clip_url` puede ser `null` inicialmente: vuelve a consultar mientras se recodifica el clip. `GET /history`, `GET /history/{id}` y `POST /compare` usan el mismo header. El JSON trae puntos del cuerpo normalizados de 0 a 1, puntaje y URLs para imágenes y clips.

Por defecto se analiza **todo el video, desde el segundo 0 hasta el final**, buscando los tiros a lo largo de la grabación. Deja vacíos `trim_start_s` y `trim_end_s`; solo complétalos si quieres recortar. No hay límite de un minuto. El límite de tamaño sigue siendo 200 MiB y se puede ampliar con `BALLIT_MAX_BYTES`.

Configuración opcional del servidor:

| Variable | Uso |
|---|---|
| `BALLIT_DATA_DIR` | Carpeta raíz de videos, resultados y caché; por defecto, este proyecto |
| `BALLIT_CORS_ORIGINS` | Orígenes permitidos separados por coma; por defecto `http://localhost:3000,http://localhost:5173` |
| `BALLIT_MAX_BYTES` | Tamaño máximo; por defecto 200 MiB |
| `BALLIT_MAX_SECONDS` | Duración máxima opcional del segmento analizado; por defecto `0` (sin límite) |
| `GROQ_API_KEY` | Habilita el coach de IA; si falta o falla, se dan consejos locales |
| `GROQ_MODEL` | Modelo de Groq; por defecto `openai/gpt-oss-20b` |

La API procesa **un video a la vez por proceso**. Ejecuta Uvicorn con un solo worker para conservar la cola y sus estados en memoria. Los resultados completos quedan en `resultados/{usuario}/`; los videos y clips en `videos_prueba/{usuario}/`. Un UUID sin login separa datos del prototipo, pero no es autenticación suficiente para un servicio público.

Para borrar videos antiguos sin borrar el historial, programa una ejecución mensual de:

```bash
python3 limpiar_videos.py --days 30
```

También admite `--max-gb` para limitar el espacio. Los directorios de datos nuevos están en `.gitignore`. Los videos que **ya estén versionados en Git** no desaparecen de su historial por añadir `.gitignore`; revisen y retiren esos archivos por separado antes de publicar el repositorio.

## ¿Qué pueden probar?

- **Analizar un video:** detectar movimientos candidatos a tiro, medir el codo y ver una imagen con el brazo marcado en verde.
- **Comparar antes y después:** ver cómo cambia la medición entre dos videos, con una tabla, un gráfico y las imágenes de los tiros.
- **Historial:** guardar resultados durante la sesión y descargarlos en CSV. El historial se mantiene en la sesión de Streamlit; todavía no hay una base de datos persistente.

Los resultados pueden aparecer como **bueno**, **dudoso** o **malo**, según la regla de referencia. Es una evaluación de la posición del codo, no de toda la técnica ni de si el balón entró en la canasta.

## 1. Descarguen el proyecto

Necesitan Git y Python. Para esta versión, usen **Python 3.10, 3.11 o 3.12**; las verificaciones de esta versión se ejecutaron con Python 3.12.

```bash
git clone https://github.com/JuandiSolo/BallItMVP.git
cd BallItMVP
```

Si ya tienen el proyecto descargado, abran una terminal dentro de su carpeta. Allí deben estar `app.py`, `analyze_elbow.py`, `experiment.py`, `requirements.txt` y `src/`.

## 2. Instalen las dependencias

En macOS o Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, desde PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Los comandos siguientes usan `python3` para macOS/Linux. En Windows, usen `python` con el entorno activado.

**Instalen desde `requirements.txt`.** La versión de Streamlit está limitada a `<1.50` para conservar la compatibilidad con MediaPipe 0.10.14.

## 3. Generen la regla con los ejemplos incluidos

Desde la carpeta principal del proyecto:

```bash
python3 analyze_elbow.py --videos videos/frente/ --test videos/sujetos/ --arm right
```

Este comando analiza los videos etiquetados de frente y calcula una regla de referencia. Después aplica esa regla a los videos de otra persona, que no tienen etiquetas.

Guarda los resultados en `elbow_out/`:

| Archivo | Qué contiene |
|---|---|
| `rule.json` | La regla que leerá la app |
| `reporte_codo.md` | Métricas, resultados y frames detectados |
| `elbow_features.csv` | Mediciones de los videos etiquetados |
| `sujetos_prueba.csv` | Mediciones de la persona de prueba |
| `hoja_pico_codo.jpg` | Momento de mayor apertura del codo |
| `hoja_release.jpg` | Momento estimado del lanzamiento |
| `hoja_sujetos_prueba.jpg` | Imágenes de los videos de prueba |

La primera extracción puede tardar. Las siguientes ejecuciones reutilizan las coordenadas guardadas en caché cuando están disponibles.

Si omiten este paso, la app usa una **regla de ejemplo** y muestra un aviso. Para reproducir la prueba actual, generen primero `rule.json`.

## 4. Abran la app

```bash
python3 -m streamlit run app.py
```

Streamlit muestra una dirección local en la terminal, normalmente:

**http://localhost:8501**

Ábranla en el navegador si no se abre automáticamente. Mantengan la terminal abierta mientras usan la app. Para detenerla, presionen **Ctrl+C**.

Para volver a abrirla otro día, activen el entorno y ejecuten el mismo comando desde la carpeta del proyecto.

## 5. Prueba rápida con dos videos

En **«Analizar un video»**, seleccionen **Derecha** y suban estos archivos por separado:

| Ejemplo | Archivo |
|---|---|
| Etiquetado como bueno | `videos/frente/bueno/42242067-8455-4e73-8727-fce7f0ba6168.mov` |
| Etiquetado con codo abierto | `videos/frente/codo/2a22ac35-6c76-4db6-9b16-f871c1630510.mov` |

Revisen que aparezcan el tiro detectado, la medición, el resultado y una imagen con el brazo que lanza en verde. Si el brazo marcado no corresponde al que tira, revisen la selección de mano.

El segundo ejemplo puede quedar **dudoso**: estar cerca del límite no equivale a un error de ejecución de la app. Estos videos también se usan para calcular la regla; esta prueba comprueba el funcionamiento, no la precisión con personas nuevas.

### Comparar los dos videos

1. Abran **«Comparar antes y después»**.
2. En **ANTES**, suban el ejemplo de codo abierto.
3. En **DESPUÉS**, suban el ejemplo bueno.
4. Revisen la diferencia numérica, la tabla, el gráfico y las dos imágenes.
5. Inviertan los archivos para comprobar el cambio en la otra dirección.

Es una demostración con dos ejemplos. Para evaluar progreso real, graben a la **misma persona**, con la **misma cámara y encuadre**, y comparen varios tiros por lado.

### Ver el historial

Usen **«Guardar este análisis en el historial»** o **«Guardar ANTES y DESPUÉS en el historial»**. Después abran la pestaña **«Historial»** para ver la tabla y descargar el CSV.

## Cómo grabar sus propios videos

- Cámara fija, de frente al jugador, aproximadamente a la altura del pecho.
- Una sola persona, con el cuerpo completo y el brazo que lanza visibles.
- Buena iluminación.
- Incluyan la preparación, el lanzamiento y un poco del movimiento posterior.
- Para empezar, prueben un video corto con un tiro. También pueden probar varios tiros en un video.

La app admite `.mp4`, `.mov`, `.m4v`, `.avi` y `.mkv`, siempre que OpenCV pueda decodificarlos. El cargador tiene un límite predeterminado de 200 MB por archivo.

**Usen los videos de `videos/frente/` para esta demo.** Las carpetas `videos/tripode/`, `videos/handheld/` y `videos/piso/` pertenecen a otro experimento de perspectiva; sus resultados no validan la regla frontal.

## Cómo leer las mediciones

La regla puede elegir distintas métricas. La app muestra la unidad correspondiente:

| Métrica | Unidad | Qué mide |
|---|---|---|
| `flare_*` | Anchos de hombro | Separación lateral del codo respecto al hombro |
| `elbow_vs_wrist_*` | Anchos de hombro | Separación lateral del codo respecto a la muñeca |
| `abduction_*` | Grados | Ángulo 2D formado por cadera, hombro y codo |

En la app, la imagen muestra un momento con el brazo elevado. La medición principal resume la preparación anterior; no representa necesariamente el valor del frame de la imagen. Las hojas generadas por `analyze_elbow.py` siguen mostrando el pico de `flare` y el release por separado.

El control **«Cambio mínimo para comparar»** usa la unidad de la regla actual. Es una sensibilidad provisional, no un umbral de mejora deportiva validado.

## Lo que todavía estamos comprobando

- La regla se calibró con pocos videos etiquetados de una persona. Falta validarla con personas nuevas.
- La perspectiva y la rotación del cuerpo pueden alterar las mediciones 2D.
- El lanzamiento se estima a partir de la muñeca; todavía no se detecta la separación entre balón y mano.
- Levantar los brazos puede confundirse con un tiro porque todavía no hay detector de balón.
- La indicación de pausa es una heurística. Tirar en un movimiento continuo no demuestra por sí solo mala técnica.
- El porcentaje leave-one-out del reporte actual deja fuera un video para aprender el umbral, pero selecciona la métrica con el conjunto completo. No debe presentarse como precisión validada con usuarios nuevos.

Los videos se procesan localmente. La app guarda las subidas y las coordenadas en `app_cache/`; no necesita una API de IA para funcionar.

## Si algo falla

**No existe una carpeta:** ejecuten los comandos desde la carpeta principal `BallItMVP` y comprueben que estén `videos/frente/` y `videos/sujetos/`.

**No se encuentra `app.py`:** necesitan la versión del proyecto que incluye la interfaz y los scripts actualizados.

**Error de MediaPipe relacionado con protobuf o `FieldDescriptor`:** con el entorno activado, reinstalen las versiones compatibles:

```bash
python3 -m pip install "streamlit>=1.40,<1.50" "protobuf>=4.25.3,<5"
```

**No detecta un tiro o marca el brazo equivocado:** revisen la mano seleccionada, el encuadre y la vista frontal. Debe verse la subida completa, con muñeca por encima del hombro y codo cerca de su altura. Si no se confirma esa postura, la app avisa y deja el video sin calificar.

**Cambiaron `app.py`:** detengan Streamlit con Ctrl+C y vuelvan a abrirlo.

## Otras herramientas del proyecto

Para revisar los momentos detectados de los videos incluidos:

```bash
python3 analyze_elbow.py --check videos/frente/ --arm right --output elbow_check
```

Para detectar tiros en el video largo y guardar clips:

```bash
python3 analyze_elbow.py --find-shots videos/release_check/largo.mp4 --arm right --cut --output elbow_long
```

Para ejecutar el pipeline original, que genera un video con esqueleto, ángulos, CSV, JSON y resumen:

```bash
python3 main.py --input sample_data/IMG_3055.MOV --output output --label "tiro libre"
```

`experiment.py` conserva el experimento de ángulos y perspectiva. `analyze_elbow.py` mide el codo de frente y genera la regla. `app.py` es la interfaz para probarlo ahora.

---

# Documentación técnica original

La guía de arriba explica cómo probar la app actual. A continuación se conserva la documentación original del pipeline y su arquitectura. Describe la primera versión; algunas limitaciones y próximos pasos ya cambiaron con la app actual.

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
