# Analisis del codo antes del tiro

## Videos etiquetados (10): {'bueno': 5, 'codo': 5}

### Que metricas separan bueno de malo

AUC: 0.5 = azar, 1.0 = separa perfecto. Unidades: flare y elbow_vs_wrist en anchos de hombro; abduction en grados.

| metrica                  | bueno (media±sd)   | malo (media±sd)   |   AUC | malo es mayor   |
|:-------------------------|:-------------------|:------------------|------:|:----------------|
| abduction_prep_mean      | 16.06±4.28         | 44.60±10.73       |     1 | True            |
| flare_prep_max           | 0.15±0.05          | 0.40±0.09         |     1 | True            |
| abduction_prep_max       | 33.62±19.41        | 90.68±13.50       |     1 | True            |
| abduction_pre_mean       | 37.08±7.45         | 81.85±17.32       |     1 | True            |
| elbow_vs_wrist_lift_mean | 0.15±0.03          | 0.30±0.06         |     1 | True            |
| flare_pre_mean           | 0.12±0.04          | 0.36±0.10         |     1 | True            |
| flare_lift_mean          | 0.18±0.04          | 0.37±0.09         |     1 | True            |
| flare_pre_max            | 0.23±0.09          | 0.48±0.10         |     1 | True            |

### Regla (una sola metrica)

- `abduction_prep_mean` < 26.30 = tiro bueno. Con todos los videos acierta 100%.
- **Leave-one-out** (el umbral se aprende sin el video que se evalua): **90%** (decir siempre 'malo': 50%).
- Zona dudosa: ±4.08 alrededor del umbral.

_Ojo: con 5 vs 5 videos, una metrica que no sirve de nada saca AUC 1.0 el 0.8% de las veces, y aqui se miraron 21. Por eso lo que cuenta es que varias formas distintas de medir lo mismo coincidan, y que se repita con otras personas._

Videos CON pausa en el set point: buenos 5 de 5, malos 2 de 5. _Si los malos tiran mas de corrido, parte de lo que se separa puede ser el ritmo y no el codo; por eso la regla usa solo geometria del codo._

### Frames detectados por video

| video                                    | label   |   start_frame |   set_frame |   release_frame |   peak_frame | has_pause   |
|:-----------------------------------------|:--------|--------------:|------------:|----------------:|-------------:|:------------|
| 42242067-8455-4e73-8727-fce7f0ba6168.mov | bueno   |            11 |          20 |              26 |           24 | True        |
| a3a900a2-42c3-4fb4-9530-ee7cdf93f82f.mov | bueno   |            19 |          28 |              31 |           29 | True        |
| afe3b83d-19a2-458b-8ec4-9da57d11d41b.mov | bueno   |             6 |          14 |              20 |           17 | True        |
| b4735dea-abd8-4d65-88f1-1c8ee163e59e.mov | bueno   |             7 |          15 |              21 |           19 | True        |
| b6c44d99-f535-4aa6-a1a4-f64e02b43a76.mov | bueno   |             6 |          15 |              19 |           18 | True        |
| 2a22ac35-6c76-4db6-9b16-f871c1630510.mov | codo    |             3 |          12 |              15 |           11 | True        |
| 2e429e85-89bc-4991-8d85-da4e3dab8851.mov | codo    |            17 |          29 |              29 |           25 | False       |
| 840e904e-edee-4354-947c-214cd59b6d23.mov | codo    |            14 |          26 |              26 |           19 | False       |
| 9e939045-0af7-40cb-9f9e-bf1da24d2dc4.mov | codo    |            16 |          25 |              28 |           27 | True        |
| a5f94a74-d7d4-4a8c-9726-04a01e0c9f5d.mov | codo    |            11 |          24 |              24 |           19 | False       |

_3 de 10 videos no tienen pausa clara en el set point (set y release casi en el mismo frame, tiro continuo): en esos, las metricas `*_set` y `*_pre_*` no son confiables; usa las `*_lift_*` (por ejemplo `flare_lift_max`) y la imagen del pico del codo._

Imagenes (linea cian = vertical del hombro): `hoja_pico_codo.jpg`, `hoja_release.jpg`. `hoja_pico_codo.jpg` muestra el momento en que el codo estuvo MAS abierto durante el levantamiento; `hoja_release.jpg` el release detectado.

## Otras personas (sin etiquetas)

Regla: `abduction_prep_mean` < 26.30 = bueno (dudoso: ±4.08). Referencia de tus videos: buenos 16.06±4.28, malos 44.60±10.73.

| video                           |   abduction_prep_mean |   flare_lift_mean |   abduction_lift_mean | has_pause   | veredicto   |
|:--------------------------------|----------------------:|------------------:|----------------------:|:------------|:------------|
| VIDEO-2026-09-30-14-34-34 2.mp4 |                 43.1  |              0.23 |                 43.1  | False       | malo        |
| VIDEO-2026-09-30-14-34-34 3.mp4 |                 33.28 |              0.46 |                 48.7  | True        | malo        |
| VIDEO-2026-09-30-14-34-34 4.mp4 |                 48.07 |              0.41 |                 53.95 | False       | malo        |
| VIDEO-2026-09-30-14-34-34 5.mp4 |                 24.18 |              0.29 |                 41.59 | True        | dudoso      |
| VIDEO-2026-09-30-14-34-34.mp4   |                 46.6  |              0.43 |                 46.6  | False       | malo        |

Imagen (momento de mayor apertura del codo): `hoja_sujetos_prueba.jpg`
