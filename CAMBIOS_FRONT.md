# BallIt — Cambios en el FRONT (repo `BallItFd`, Angular 20, desplegado en Vercel)

> **Para Claude Code:** este archivo es tu única fuente para el front. El back y su **contrato de demos** están en `CAMBIOS_BACK.md` (sección 2); léelo primero, porque aquí solo se consume. **[HECHO VERIFICADO]** = lo confirmé leyendo el código.

---

## 0. Qué se quiere

En la app, una opción **"Prueba con un video de ejemplo"** (3 videos que el back ya analizó). El recorrido es:

1. El usuario elige un ejemplo → ve el **video original** (con un botón **"Analizar este tiro"**).
2. Pulsa el botón → pantalla de **"Analizando…" simulada** (~5 s, barra y pasos, igual que un análisis real).
3. Aparece la pantalla de **Resultado de siempre**, pero reproduciendo el **video anotado** (esqueleto + ángulo ya dibujados) **junto con todos sus datos**: puntaje, veredicto, coach, gráfica, tiros.

Los demos **no** se guardan en el historial, no cuentan para la racha ni se pueden comparar.

---

## 1. Hallazgos

| # | Hallazgo |
|---|---|
| F-H1 | **[HECHO VERIFICADO]** Este repo ya apunta al back desplegado: `environment.apiBase = 'https://ballit-api.onrender.com'`. **No tocar `environment.ts` ni crear environments nuevos.** (Para probar contra un back local, cambiar `apiBase` temporalmente **sin commitearlo**.) |
| F-H2 | **[HECHO VERIFICADO]** `AnalysisVideo` hoy descarga el video como *blob* con `X-User-Id` y dibuja el esqueleto en un `<canvas>` con `result.frames`. Para el demo se usará el **video anotado** y no se dibuja nada encima. |
| F-H3 | **[HECHO VERIFICADO]** `SecureMedia` y `AnalysisVideo` llaman a `api.mediaUrl(path)` y hacen `URL.revokeObjectURL` al limpiar. Si `mediaUrl` devuelve una URL normal, el `revoke` es un no-op inofensivo. |
| F-H4 | **[HECHO VERIFICADO]** Las clases del bloque "Analizando…" (`.processing`, `.ball-wrap`, `.ball`, `.bar`, `.bar-fill`, `.tip-now`, `.skeleton`) son **globales** (`styles.scss`), así que una página nueva las puede reutilizar sin copiar CSS. |
| F-H5 | **[HECHO VERIFICADO]** `ResultView` recibe `result` y `mediaPending`; todo lo demás (anillo de puntaje, gráfica, coach, carrusel de tiros) sale del JSON. Los clips `shotN.mp4` y las imágenes `shotN.jpg` los muestra `SecureMedia` con `/files/...`; en demo serán `/demos/...`. |
| F-H6 | **[SUPUESTO]** El back está en Render. En un plan gratuito **se duerme**: la primera petición puede tardar 30–60 s. La UI debe tolerarlo (F6). |

---

## 2. Cambios

### F1. Modelos (`core/models.ts`)

```ts
export interface DemoMeta {
  slug: string; title: string; description: string;
  thumb_url: string; video_url: string; annotated_url: string;
}
export interface DemoListItem extends DemoMeta {
  duration_s: number; n_shots: number; score: number; verdict: Verdict;
}
// en AnalysisResult agregar:   demo?: DemoMeta;
```

### F2. `BallitApi` (`core/ballit-api.service.ts`)

1. `demos(): Observable<DemoListItem[]>` → `GET {base}/demos`.
2. `demo(slug): Observable<AnalysisResult>` → `GET {base}/demos/{slug}`.
3. `abs(path: string): string` → `base + path` (para `<img>`/`<video>` de demos).
4. **`mediaUrl(path)`**: si `path.startsWith('/demos/')` → `return of(this.base + path)` (URL directa: el `<video>` hace *streaming* y *seek*, y Safari recibe `Range`). Para lo demás, comportamiento actual intacto. Con esto `SecureMedia` y los clips del carrusel funcionan sin cambios (F-H3).
5. `simulateProcessing(): Observable<{ progress: number; step: string }>` — emite cada ~200 ms durante **~5 s** (≈1.5 s si `matchMedia('(prefers-reduced-motion: reduce)').matches`), progreso suavizado 0 → 1, con **los mismos textos que el back real**: `'Preparando video'` (0–5 %), `'Detectando movimiento'` (5–80 %), `'Generando feedback'` (80–100 %). Completa al llegar a 1.

### F3. Interceptor (`core/user-id.interceptor.ts`)

No agregar `X-User-Id` a URLs que empiecen con `${environment.apiBase}/demos`. Así las peticiones demo son "simples" para CORS (sin *preflight*) y no mezclan los demos con datos de usuario.

### F4. Ruta y página nueva `DemoPage`

`app.routes.ts`:

```ts
{ path: 'demo/:slug',
  loadComponent: () => import('./features/demo/demo').then((m) => m.DemoPage) },
```

`features/demo/demo.ts|html` (standalone, signals, igual que el resto). Máquina de 3 estados, `phase = signal<'preview' | 'processing' | 'result'>('preview')`:

- **Carga inicial:** `api.demo(slug)` → `result` (también sirve como precarga: mientras el usuario mira el video original, el JSON ya viene bajando). Si falla: bloque de error como el de `AnalysisPage` con `toApiError` y botón "Volver a los ejemplos" (`/subir`).
- **`preview`:** `<app-page-header title="Ejemplo" back="/subir" />`, el **video original** `<video [src]="api.abs(demo.video_url)" [poster]="api.abs(demo.thumb_url)" controls playsinline muted preload="metadata">` (sin canvas), título y descripción del demo, y un botón grande **"Analizar este tiro"** (`btn primary block`, dentro de `cta-bar`) que pone `phase='processing'`. Sin tarjeta de veredicto aquí: el resultado se "descubre" después.
- **`processing`:** mismo marcado que `analysis.html` (bola, `{{ step }}`, barra, `%`, tarjeta "💡 Mientras esperas" con los mismos 4 tips) alimentado por `api.simulateProcessing()`. Al completar y tener `result` → `phase='result'`. Si el JSON aún no llegó, quedarse en 100 % mostrando "Generando feedback" hasta que llegue (no falla).
- **`result`:** `<app-page-header title="Resultado" back="/subir" />` + `<app-result-view [result]="r" [mediaPending]="false" />`. Hacer `scrollTo(0)` al entrar.

### F5. `AnalysisVideo` en modo demo (`features/analysis/analysis-video.ts`)

Cuando `result().demo?.annotated_url` existe:
- `path` = `demo.annotated_url` (en vez de `video.url`). `mediaUrl` ya lo resuelve como URL directa (F2.4).
- **No dibujar el canvas** (`redraw()` retorna tras limpiar) y **ocultar los toggles** Esqueleto/Ángulo/Estela (el esqueleto ya está en el video). **Mantener** el selector de velocidad (cámara lenta) y `seek()`: los botones de "momentos clave" siguen funcionando porque el anotado dura igual que el original (regla de oro del back).
- Sin `demo`, comportamiento actual **idéntico** (el spec `pose-overlay.spec.ts` debe seguir pasando).

### F6. Selección de ejemplos (`features/upload/upload.ts|html`)

Debajo del dropzone y antes de la tarjeta de opciones, nueva sección **"¿No tienes un video a mano? Prueba con uno de ejemplo"** con 3 tarjetas: miniatura (`api.abs(thumb_url)`), título, descripción, y una etiqueta de veredicto con las clases `badge bueno|dudoso|malo` que ya existen. Cada tarjeta es `routerLink="['/demo', d.slug]"`.
- Cargar con `api.demos()`. Mientras carga: 3 `.skeleton`. Si pasan **> 4 s**, mostrar "Despertando el servidor, un momento…" (F-H6).
- Si falla: **ocultar la sección en silencio** (la subida real sigue funcionando).
- `features/home/home.html`: en el estado vacío (junto a "Analizar mi primer tiro") añadir un enlace secundario "Ver un ejemplo" → `/subir`.

### F7. Banner de ejemplo en `result-view.html`

Si `r.demo`, mostrar arriba (antes del video) una nota discreta: **"Este es un análisis de ejemplo."** con botón **"Analizar mi propio video"** → `/subir`. Dejar los botones existentes. En modo demo ocultar "Ver historial" (no hay nada guardado).

### F8. Lo que NO debe pasar

- Los demos no pasan por `/analyze` ni por el historial; `home`, `history` y `streak` dependen de `api.history()`, así que no hay nada que hacer salvo **no** romperlo.
- `compare` no ofrece demos. No modificar `compare` ni `profile`.
- No tocar `environment.ts`, `angular.json` ni el flujo real de subida.

### F9. Tests (el repo ya usa `*.spec.ts`)

- `simulateProcessing()` con `fakeAsync`: emite pasos en orden y completa en 1.
- `mediaUrl('/demos/x/files/video.mp4')` devuelve la URL directa y no lanza petición HTTP; `mediaUrl('/files/...')` mantiene el comportamiento anterior.
- `DemoPage`: pasa `preview → processing → result` al pulsar el botón (con `api` simulado).
- Los specs actuales (`result-view.spec.ts`, `pose-overlay.spec.ts`, `streak.spec.ts`) siguen pasando.

---

## 3. Checks de aceptación

- [ ] `/subir` muestra 3 tarjetas con miniatura y veredicto; también funcionan con el back dormido (aparece el aviso y luego cargan).
- [ ] Al elegir una: se ve el **video original**; al pulsar "Analizar este tiro" → ~5 s de progreso → resultado con **video anotado**, puntaje, veredicto, coach (el texto de Groq ya guardado), gráfica y carrusel de tiros con clips.
- [ ] En el resultado demo no hay toggles de esqueleto; la velocidad 0.5×/0.25× funciona; tocar un "momento clave" salta al instante correcto del anotado.
- [ ] **Celular real** (Safari iOS y Chrome Android): ambos videos se reproducen y se puede hacer *seek*.
- [ ] No aparece nada en Historial ni cambia la racha tras ver un demo.
- [ ] Subir un video propio sigue funcionando igual.
- [ ] `ng build --configuration production` sin romper los presupuestos de `angular.json` (500 kB warning / 1 MB error).

## 4. Dependencias entre archivos

- Esto requiere que el back ya tenga `/demos` desplegado y CORS con el dominio del front (`CAMBIOS_BACK.md`, B4.1).
- Orden: back (con demos generados) → front → landing.
