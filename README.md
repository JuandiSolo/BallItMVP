🏀 BallIt MVP

¡Hey! Si quieren probar lo que llevamos de BallIt, pueden correrlo en su computador y abrir la app en el navegador. Suben un video de un tiro libre grabado de frente, revisan la posición del codo antes de lanzar y comparan dos videos.

Por ahora es un prototipo local. Esta guía explica cómo instalarlo, verlo y probarlo con los videos incluidos.

¿Qué pueden probar?

• Analizar un video: detectar movimientos candidatos a tiro, medir el codo y ver una imagen con el brazo marcado en verde.
• Comparar antes y después: ver cómo cambia la medición entre dos videos, con una tabla, un gráfico y las imágenes de los tiros.
• Historial: guardar resultados durante la sesión y descargarlos en CSV. El historial se mantiene en la sesión de Streamlit; todavía no hay una base de datos persistente.

Los resultados pueden aparecer como bueno, dudoso o malo, según la regla de referencia. Es una evaluación de la posición del codo, no de toda la técnica ni de si el balón entró en la canasta.

1. Descarguen el proyecto

Necesitan Git y Python. Para esta versión, usen Python 3.10, 3.11 o 3.12; las verificaciones de esta versión se ejecutaron con Python 3.12.

git clone https://github.com/JuandiSolo/BallItMVP.git
cd BallItMVP

Si ya tienen el proyecto descargado, abran una terminal dentro de su carpeta. Allí deben estar app.py, analyze_elbow.py, experiment.py, requirements.txt y src/.

2. Instalen las dependencias

En macOS o Linux:

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

En Windows, desde PowerShell:

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

Los comandos siguientes usan python3 para macOS/Linux. En Windows, usen python con el entorno activado.

Instalen desde requirements.txt. La versión de Streamlit está limitada a <1.50 para conservar la compatibilidad con MediaPipe 0.10.14.

3. Generen la regla con los ejemplos incluidos

Desde la carpeta principal del proyecto:

python3 analyze_elbow.py --videos videos/frente/ --test videos/sujetos/ --arm right

Este comando analiza los videos etiquetados de frente y calcula una regla de referencia. Después aplica esa regla a los videos de otra persona, que no tienen etiquetas.

Guarda los resultados en elbow_out/:

|Archivo                  |Qué contiene                            |
|-------------------------|----------------------------------------|
|`rule.json`              |La regla que leerá la app               |
|`reporte_codo.md`        |Métricas, resultados y frames detectados|
|`elbow_features.csv`     |Mediciones de los videos etiquetados    |
|`sujetos_prueba.csv`     |Mediciones de la persona de prueba      |
|`hoja_pico_codo.jpg`     |Momento de mayor apertura del codo      |
|`hoja_release.jpg`       |Momento estimado del lanzamiento        |
|`hoja_sujetos_prueba.jpg`|Imágenes de los videos de prueba        |

La primera extracción puede tardar. Las siguientes ejecuciones reutilizan las coordenadas guardadas en caché cuando están disponibles.

Si omiten este paso, la app usa una regla de ejemplo y muestra un aviso. Para reproducir la prueba actual, generen primero rule.json.

4. Abran la app

python3 -m streamlit run app.py

Streamlit muestra una dirección local en la terminal, normalmente:

http://localhost:8501

Ábranla en el navegador si no se abre automáticamente. Mantengan la terminal abierta mientras usan la app. Para detenerla, presionen Ctrl+C.

Para volver a abrirla otro día, activen el entorno y ejecuten el mismo comando desde la carpeta del proyecto.

5. Prueba rápida con dos videos

En «Analizar un video», seleccionen Derecha y suban estos archivos por separado:

|Ejemplo                    |Archivo                                                       |
|---------------------------|--------------------------------------------------------------|
|Etiquetado como bueno      |`videos/frente/bueno/42242067-8455-4e73-8727-fce7f0ba6168.mov`|
|Etiquetado con codo abierto|`videos/frente/codo/2a22ac35-6c76-4db6-9b16-f871c1630510.mov` |

Revisen que aparezcan el tiro detectado, la medición, el resultado y una imagen con el brazo que lanza en verde. Si el brazo marcado no corresponde al que tira, revisen la selección de mano.

El segundo ejemplo puede quedar dudoso: estar cerca del límite no equivale a un error de ejecución de la app. Estos videos también se usan para calcular la regla; esta prueba comprueba el funcionamiento, no la precisión con personas nuevas.

Comparar los dos videos

1. Abran «Comparar antes y después».
2. En ANTES, suban el ejemplo de codo abierto.
3. En DESPUÉS, suban el ejemplo bueno.
4. Revisen la diferencia numérica, la tabla, el gráfico y las dos imágenes.
5. Inviertan los archivos para comprobar el cambio en la otra dirección.

Es una demostración con dos ejemplos. Para evaluar progreso real, graben a la misma persona, con la misma cámara y encuadre, y comparen varios tiros por lado.

Ver el historial

Usen «Guardar este análisis en el historial» o «Guardar ANTES y DESPUÉS en el historial». Después abran la pestaña «Historial» para ver la tabla y descargar el CSV.

Cómo grabar sus propios videos

• Cámara fija, de frente al jugador, aproximadamente a la altura del pecho.
• Una sola persona, con el cuerpo completo y el brazo que lanza visibles.
• Buena iluminación.
• Incluyan la preparación, el lanzamiento y un poco del movimiento posterior.
• Para empezar, prueben un video corto con un tiro. También pueden probar varios tiros en un video.

La app admite .mp4, .mov, .m4v, .avi y .mkv, siempre que OpenCV pueda decodificarlos. El cargador tiene un límite predeterminado de 200 MB por archivo.

Usen los videos de videos/frente/ para esta demo. Las carpetas videos/tripode/, videos/handheld/ y videos/piso/ pertenecen a otro experimento de perspectiva; sus resultados no validan la regla frontal.

Cómo leer las mediciones

La regla puede elegir distintas métricas. La app muestra la unidad correspondiente:

|Métrica           |Unidad          |Qué mide                                        |
|------------------|----------------|------------------------------------------------|
|`flare_*`         |Anchos de hombro|Separación lateral del codo respecto al hombro  |
|`elbow_vs_wrist_*`|Anchos de hombro|Separación lateral del codo respecto a la muñeca|
|`abduction_*`     |Grados          |Ángulo 2D formado por cadera, hombro y codo     |

La imagen del pico muestra flare max, aunque la regla use otra métrica. Por eso el número sobre la imagen y el número principal pueden ser diferentes.

El control «Cambio mínimo para comparar» usa la unidad de la regla actual. Es una sensibilidad provisional, no un umbral de mejora deportiva validado.

Lo que todavía estamos comprobando

• La regla se calibró con pocos videos etiquetados de una persona. Falta validarla con personas nuevas.
• La perspectiva y la rotación del cuerpo pueden alterar las mediciones 2D.
• El lanzamiento se estima a partir de la muñeca; todavía no se detecta la separación entre balón y mano.
• Levantar los brazos puede confundirse con un tiro porque todavía no hay detector de balón.
• La indicación de pausa es una heurística. Tirar en un movimiento continuo no demuestra por sí solo mala técnica.
• El porcentaje leave-one-out del reporte actual deja fuera un video para aprender el umbral, pero selecciona la métrica con el conjunto completo. No debe presentarse como precisión validada con usuarios nuevos.

Los videos se procesan localmente. La app guarda las subidas y las coordenadas en app_cache/; no necesita una API de IA para funcionar.

Si algo falla

No existe una carpeta: ejecuten los comandos desde la carpeta principal BallItMVP y comprueben que estén videos/frente/ y videos/sujetos/.

No se encuentra app.py: necesitan la versión del proyecto que incluye la interfaz y los scripts actualizados.

Error de MediaPipe relacionado con protobuf o FieldDescriptor: con el entorno activado, reinstalen las versiones compatibles:

python3 -m pip install "streamlit>=1.40,<1.50" "protobuf>=4.25.3,<5"

No detecta un tiro o marca el brazo equivocado: revisen la mano seleccionada, el encuadre y la vista frontal. Si no consigue detectar un tiro claro, puede intentar analizar el clip completo y mostrar un aviso.

Cambiaron app.py: detengan Streamlit con Ctrl+C y vuelvan a abrirlo.

Otras herramientas del proyecto

Para revisar los momentos detectados de los videos incluidos:

python3 analyze_elbow.py --check videos/frente/ --arm right --output elbow_check

Para detectar tiros en el video largo y guardar clips:

python3 analyze_elbow.py --find-shots videos/release_check/largo.mp4 --arm right --cut --output elbow_long

Para ejecutar el pipeline original, que genera un video con esqueleto, ángulos, CSV, JSON y resumen:

python3 main.py --input sample_data/IMG_3055.MOV --output output --label "tiro libre"

experiment.py conserva el experimento de ángulos y perspectiva. analyze_elbow.py mide el codo de frente y genera la regla. app.py es la interfaz para probarlo ahora.