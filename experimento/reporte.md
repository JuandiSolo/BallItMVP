# Resultados del experimento de tiro libre

## Camara: tripode  (15 videos)

Videos por clase: {'brazo': 5, 'bueno': 5, 'codo': 5}

Deteccion de persona (promedio): 100.0%  (minimo 100.0%)

### Metricas que mejor separan bueno de malo

AUC: 0.5 = no separa nada (azar), 1.0 = separa perfecto.

| metrica       | bueno (media±sd)   | codo (media±sd)   |   AUC vs codo | brazo (media±sd)   |   AUC vs brazo |
|:--------------|:-------------------|:------------------|--------------:|:-------------------|---------------:|
| elbow_rel     | 173.2±3.0          | 150.7±8.8         |          1    | 169.1±1.4          |           0.92 |
| knee_range    | 54.4±4.1           | 57.6±2.5          |          0.8  | 46.6±4.3           |           0.96 |
| knee_min      | 123.9±4.6          | 120.4±1.2         |          0.8  | 131.7±5.2          |           0.88 |
| hip_mean      | 168.7±1.7          | 169.7±0.6         |          0.72 | 171.7±1.1          |           0.96 |
| shoulder_mean | 84.0±22.5          | 102.5±5.5         |          0.76 | 110.6±10.1         |           0.88 |
| hip_range     | 18.2±4.7           | 20.1±1.6          |          0.64 | 9.5±1.8            |           0.96 |
| elbow_min     | 40.2±16.4          | 72.7±12.2         |          0.92 | 37.3±7.4           |           0.56 |
| knee_mean     | 156.7±4.2          | 157.9±1.4         |          0.6  | 162.7±2.3          |           0.88 |

### Reglas candidatas (una sola metrica + umbral)

- **Error de codo**: `elbow_rel` > 163.2° = tiro bueno (acierta 100% de 10 videos)
- **Error de brazo**: `shoulder_mean` < 105.5° = tiro bueno (acierta 90% de 10 videos)

### Clasificador simple (leave-one-out)

- Acierto bueno / codo / brazo: **87%**
- Acierto bueno vs malo: **87%**
- Referencia (decir siempre 'malo'): 67%

Matriz de confusion (filas = real, columnas = predicho):

| real   |   bueno |   codo |   brazo |
|:-------|--------:|-------:|--------:|
| bueno  |       3 |      1 |       1 |
| codo   |       0 |      5 |       0 |
| brazo  |       0 |      0 |       5 |

Videos mal clasificados:

| video | real | predicho | tipo de error |
|:--|:--|:--|:--|
| IMG_3492.mov | bueno | brazo | bueno <-> malo |
| IMG_3496.mov | bueno | codo | bueno <-> malo |

Grafico: `boxplot_tripode.png`

## Camara: handheld  (14 videos)

Videos por clase: {'brazo': 5, 'bueno': 5, 'codo': 4}

Deteccion de persona (promedio): 100.0%  (minimo 100.0%)

### Metricas que mejor separan bueno de malo

AUC: 0.5 = no separa nada (azar), 1.0 = separa perfecto.

| metrica        | bueno (media±sd)   | codo (media±sd)   |   AUC vs codo | brazo (media±sd)   |   AUC vs brazo |
|:---------------|:-------------------|:------------------|--------------:|:-------------------|---------------:|
| shoulder_rel   | 167.4±2.7          | 175.0±3.3         |          1    | 173.7±3.6          |           0.92 |
| shoulder_max   | 166.8±3.3          | 175.3±3.2         |          1    | 173.8±3.6          |           0.92 |
| hip_max        | 175.4±0.9          | 177.3±0.8         |          0.95 | 177.7±1.3          |           0.92 |
| hip_mean       | 169.0±1.5          | 171.2±1.3         |          0.85 | 171.8±1.3          |           0.92 |
| hip_rel        | 175.4±1.3          | 176.8±0.2         |          0.95 | 174.0±3.5          |           0.64 |
| shoulder_range | 160.3±5.4          | 163.5±5.8         |          0.7  | 167.0±3.3          |           0.84 |
| elbow_max      | 177.4±1.4          | 176.9±0.6         |          0.7  | 174.3±3.4          |           0.84 |
| shoulder_min   | 6.5±2.7            | 11.8±2.8          |          1    | 6.8±2.0            |           0.52 |

### Reglas candidatas (una sola metrica + umbral)

- **Error de codo**: `shoulder_min` < 9.3° = tiro bueno (acierta 100% de 9 videos)
- **Error de brazo**: `elbow_max` > 175.1° = tiro bueno (acierta 90% de 10 videos)

### Clasificador simple (leave-one-out)

- Acierto bueno / codo / brazo: **71%**
- Acierto bueno vs malo: **86%**
- Referencia (decir siempre 'malo'): 64%

Matriz de confusion (filas = real, columnas = predicho):

| real   |   bueno |   codo |   brazo |
|:-------|--------:|-------:|--------:|
| bueno  |       4 |      0 |       1 |
| codo   |       0 |      3 |       1 |
| brazo  |       1 |      1 |       3 |

Videos mal clasificados:

| video | real | predicho | tipo de error |
|:--|:--|:--|:--|
| IMG_3520.mov | brazo | codo | confundio el tipo de error |
| IMG_3523.mov | brazo | bueno | bueno <-> malo |
| IMG_3513.mov | bueno | brazo | bueno <-> malo |
| IMG_3516.mov | codo | brazo | confundio el tipo de error |

Grafico: `boxplot_handheld.png`

## Camara: piso  (15 videos)

Videos por clase: {'brazo': 5, 'bueno': 5, 'codo': 5}

Deteccion de persona (promedio): 100.0%  (minimo 100.0%)

### Metricas que mejor separan bueno de malo

AUC: 0.5 = no separa nada (azar), 1.0 = separa perfecto.

| metrica       | bueno (media±sd)   | codo (media±sd)   |   AUC vs codo | brazo (media±sd)   |   AUC vs brazo |
|:--------------|:-------------------|:------------------|--------------:|:-------------------|---------------:|
| hip_range     | 26.2±4.3           | 18.5±4.2          |          0.92 | 16.5±2.3           |           1    |
| shoulder_max  | 170.8±1.7          | 175.0±2.2         |          0.96 | 175.7±3.5          |           0.92 |
| hip_min       | 151.2±4.8          | 159.6±4.5         |          0.88 | 160.7±2.1          |           0.96 |
| hip_mean      | 166.7±2.9          | 170.3±1.6         |          0.88 | 169.9±0.8          |           0.84 |
| shoulder_rel  | 168.9±3.9          | 173.6±4.7         |          0.84 | 175.5±3.4          |           0.88 |
| shoulder_mean | 91.5±5.1           | 95.2±8.4          |          0.64 | 112.2±9.6          |           1    |
| knee_range    | 56.8±5.3           | 52.0±2.2          |          0.8  | 49.0±6.0           |           0.84 |
| knee_mean     | 158.3±2.8          | 161.4±2.9         |          0.8  | 161.8±2.9          |           0.8  |

### Reglas candidatas (una sola metrica + umbral)

- **Error de codo**: `shoulder_min` < 8.2° = tiro bueno (acierta 90% de 10 videos)
- **Error de brazo**: `shoulder_mean` < 99.1° = tiro bueno (acierta 100% de 10 videos)

### Clasificador simple (leave-one-out)

- Acierto bueno / codo / brazo: **73%**
- Acierto bueno vs malo: **87%**
- Referencia (decir siempre 'malo'): 67%

Matriz de confusion (filas = real, columnas = predicho):

| real   |   bueno |   codo |   brazo |
|:-------|--------:|-------:|--------:|
| bueno  |       4 |      1 |       0 |
| codo   |       1 |      3 |       1 |
| brazo  |       0 |      1 |       4 |

Videos mal clasificados:

| video | real | predicho | tipo de error |
|:--|:--|:--|:--|
| IMG_3542.mov | brazo | codo | confundio el tipo de error |
| IMG_3530.mov | bueno | codo | bueno <-> malo |
| IMG_3532.mov | codo | bueno | bueno <-> malo |
| IMG_3534.mov | codo | brazo | confundio el tipo de error |

Grafico: `boxplot_piso.png`

## Robustez: entrenar con tripode, probar con otras camaras

Si el acierto cae mucho, las metricas dependen de la perspectiva de la camara.

### tripode -> handheld

- Acierto bueno / codo / brazo: **50%**
- Acierto bueno vs malo: **71%**
- Referencia (decir siempre 'malo'): 64%

| real   |   bueno |   codo |   brazo |
|:-------|--------:|-------:|--------:|
| bueno  |       1 |      0 |       4 |
| codo   |       0 |      1 |       3 |
| brazo  |       0 |      0 |       5 |

Videos mal clasificados:

| video | real | predicho | tipo de error |
|:--|:--|:--|:--|
| IMG_3510.mov | bueno | brazo | bueno <-> malo |
| IMG_3511.mov | bueno | brazo | bueno <-> malo |
| IMG_3512.mov | bueno | brazo | bueno <-> malo |
| IMG_3513.mov | bueno | brazo | bueno <-> malo |
| IMG_3514.mov | codo | brazo | confundio el tipo de error |
| IMG_3515.mov | codo | brazo | confundio el tipo de error |
| IMG_3516.mov | codo | brazo | confundio el tipo de error |

### tripode -> piso

- Acierto bueno / codo / brazo: **33%**
- Acierto bueno vs malo: **53%**
- Referencia (decir siempre 'malo'): 67%

| real   |   bueno |   codo |   brazo |
|:-------|--------:|-------:|--------:|
| bueno  |       0 |      5 |       0 |
| codo   |       0 |      5 |       0 |
| brazo  |       2 |      3 |       0 |

Videos mal clasificados:

| video | real | predicho | tipo de error |
|:--|:--|:--|:--|
| IMG_3538.mov | brazo | codo | confundio el tipo de error |
| IMG_3540.mov | brazo | codo | confundio el tipo de error |
| IMG_3541.mov | brazo | codo | confundio el tipo de error |
| IMG_3542.mov | brazo | bueno | bueno <-> malo |
| IMG_3543.mov | brazo | bueno | bueno <-> malo |
| IMG_3526.mov | bueno | codo | bueno <-> malo |
| IMG_3527.mov | bueno | codo | bueno <-> malo |
| IMG_3528.mov | bueno | codo | bueno <-> malo |
| IMG_3529.mov | bueno | codo | bueno <-> malo |
| IMG_3530.mov | bueno | codo | bueno <-> malo |

## Diagnostico del release

Graficos por camara (azul = altura de muneca, naranja = angulo del codo; verde = release por muneca, rojo = por codo, magenta = manual, el elegido es el que se usa en las metricas *_rel):

- `debug_release_tripode.png`
- `debug_release_handheld.png`
- `debug_release_piso.png`

Videos donde muneca y codo difieren en mas de 5 frames: **13 de 44**
(los mas sospechosos, revisalos primero): IMG_3511.mov, IMG_3512.mov, IMG_3514.mov, IMG_3515.mov, IMG_3517.mov, IMG_3533.mov, IMG_3536.mov, IMG_3508.mov, IMG_3498.mov, IMG_3499.mov, IMG_3500.mov, IMG_3501.mov, IMG_3502.mov

Videos donde el release automatico cae en los primeros/ultimos 3 frames (casi seguro mal, el clip deberia tener movimiento despues del release): **1** -> IMG_3496.mov

### Error contra tus frames manuales

- Metodo **muneca**: error mediano 1.0 frames; dentro de ±3 frames: 100% de 15 videos
- Metodo **codo**: error mediano 3.0 frames; dentro de ±3 frames: 60% de 15 videos
- Metodo **elegido (metodo + offset)**: error mediano 1.0 frames; dentro de ±3 frames: 100% de 15 videos
- Sesgo mediano del metodo muneca: **-1.0 frames** (negativo = detecta antes que tu; corregirlo: `--release-offset 1`)
- Sesgo mediano del metodo codo: **-2.0 frames** (negativo = detecta antes que tu; corregirlo: `--release-offset 2`)

| video        |   manual |   muneca |   codo |   error muneca (+ = tarde) |   error codo (+ = tarde) |
|:-------------|---------:|---------:|-------:|---------------------------:|-------------------------:|
| IMG_3520.mov |       24 |       22 |     20 |                         -2 |                       -4 |
| IMG_3521.mov |       34 |       33 |     31 |                         -1 |                       -3 |
| IMG_3522.mov |       26 |       24 |     24 |                         -2 |                       -2 |
| IMG_3513.mov |       21 |       20 |     19 |                         -1 |                       -2 |
| IMG_3542.mov |       26 |       25 |     24 |                         -1 |                       -2 |
| IMG_3506.mov |       30 |       28 |     25 |                         -2 |                       -5 |
| IMG_3508.mov |       25 |       25 |     34 |                          0 |                        9 |
| IMG_3492.mov |       19 |       19 |     19 |                          0 |                        0 |
| IMG_3493.mov |       24 |       23 |     22 |                         -1 |                       -2 |
| IMG_3494.mov |       20 |       21 |     18 |                          1 |                       -2 |
| IMG_3495.mov |       26 |       25 |     23 |                         -1 |                       -3 |
| IMG_3496.mov |       27 |       27 |     29 |                          0 |                        2 |
| IMG_3500.mov |       27 |       26 |     35 |                         -1 |                        8 |
| IMG_3501.mov |       28 |       27 |     35 |                         -1 |                        7 |
| IMG_3502.mov |       30 |       28 |     40 |                         -2 |                       10 |

### Metodo muneca: efecto del umbral (--wrist-frac)

|   umbral |   error absoluto mediano | dentro de ±3 frames   |   sesgo mediano (+ = tarde) |   videos |
|---------:|-------------------------:|:----------------------|----------------------------:|---------:|
|     0.7  |                        7 | 0%                    |                          -7 |       15 |
|     0.75 |                        4 | 40%                   |                          -4 |       15 |
|     0.8  |                        3 | 53%                   |                          -3 |       15 |
|     0.85 |                        2 | 93%                   |                          -2 |       15 |
|     0.9  |                        1 | 100%                  |                          -1 |       15 |
|     0.95 |                        1 | 100%                  |                           1 |       15 |
