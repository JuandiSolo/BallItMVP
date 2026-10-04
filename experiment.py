#!/usr/bin/env python3
"""
experiment.py
--------------
Experimento para validar si el sistema REALMENTE distingue un tiro libre
bueno de uno malo, usando tus videos etiquetados.

Que hace:
  1. Recorre una carpeta de videos y deduce de la ruta/nombre la CAMARA
     (tripode / handheld / piso) y la CLASE (bueno / codo / brazo).
  2. Pasa cada video por tu pipeline (PoseExtractor + compute_all_angles)
     y guarda los angulos por frame en cache (para no reprocesar).
  3. Resume cada video en unas pocas metricas (min/max/promedio/rango de
     codo, hombro, rodilla y cadera del brazo que tira).
  4. Para cada camara compara los grupos: que metricas separan bueno de
     malo, y que tan bien clasifica un modelo muy simple.
  5. Prueba robustez: entrena con tripode y evalua con handheld / piso.

Uso:
    python experiment.py --videos videos/
    python experiment.py --videos videos/ --arm right --force

Estructura de carpetas aceptada (cualquiera de las dos funciona):
    videos/tripode/bueno/tiro_01.mov
    videos/tripode_bueno_01.mov          (todo en una carpeta, con nombres claros)
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.pose_extractor import PoseExtractor
from src.angle_calculator import landmarks_to_xy, compute_all_angles, LANDMARK_IDS

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".m4v", ".mkv"}

# Palabras clave para deducir camara y clase desde la ruta del archivo.
CAMERA_KEYS = {
    "tripode": ["tripode", "trepied", "tripod"],
    "handheld": ["handheld", "mano", "hand"],
    "piso": ["piso", "suelo", "floor"],
}
CLASS_KEYS = {
    "bueno": ["bueno", "bien", "good", "correcto"],
    "codo": ["codo", "elbow"],
    "brazo": ["brazo", "arm"],
}

JOINTS = ["elbow", "shoulder", "knee", "hip"]
MIN_DETECTED_FRAMES = 10
TOP_K_FEATURES = 4


# --------------------------------------------------------------------------
# 1. Deteccion de camara / clase
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _match(text: str, table: dict):
    for label, words in table.items():
        if any(w in text for w in words):
            return label
    return None


def find_videos(root: Path):
    """Devuelve (lista de videos reconocidos, lista de no reconocidos)."""
    found, unknown = [], []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = _norm(str(path.relative_to(root)))
        camera, label = _match(rel, CAMERA_KEYS), _match(rel, CLASS_KEYS)
        if camera and label:
            found.append({"path": path, "camera": camera, "label": label})
        else:
            unknown.append(path)
    return found, unknown


# --------------------------------------------------------------------------
# 2. Extraccion de angulos por video (con cache)
# --------------------------------------------------------------------------
def extract_angles(video_path: Path, cache_path: Path, complexity: int, force: bool):
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
        if "right_wrist_y" in cached.columns:
            return cached
        print("     (cache viejo sin coordenadas: reprocesando este video)")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    extractor = PoseExtractor(model_complexity=complexity)
    rows, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = extractor.process_frame(frame)
        row = {"frame": idx, "timestamp_s": idx / fps, "detected": False}
        if result.pose_landmarks:
            points = landmarks_to_xy(result.pose_landmarks.landmark, w, h)
            row.update(compute_all_angles(points))
            for k, pt in points.items():  # coordenadas en pixeles (para altura de muneca, tiras, etc.)
                row[f"{k}_x"], row[f"{k}_y"] = float(pt[0]), float(pt[1])
            row["detected"] = True
        rows.append(row)
        idx += 1
    cap.release()
    extractor.close()

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# --------------------------------------------------------------------------
# 3. Metricas por video
# --------------------------------------------------------------------------
def _smoothed(df: pd.DataFrame):
    """Frames con persona detectada, suavizados (mediana de 5 frames) e indexados por numero de frame."""
    det = df[df["detected"] == True]  # noqa: E712
    cols = [c for c in df.columns if c.endswith(("_angle", "_x", "_y"))]
    sm = det[cols].rolling(5, center=True, min_periods=1).median()
    sm.index = det["frame"].values
    return det, sm


def wrist_height(sm: pd.DataFrame, arm: str):
    """
    Altura de la muneca sobre el hombro, medida en 'largos de torso'
    (positivo = muneca mas arriba que el hombro). Al restar el hombro y normalizar
    por el torso, da igual el zoom o que la camara se mueva un poco.
    """
    need = [f"{arm}_{j}_{c}" for j in ("wrist", "shoulder", "hip") for c in ("x", "y")]
    if any(c not in sm.columns for c in need):
        return None
    torso = np.hypot(sm[f"{arm}_shoulder_x"] - sm[f"{arm}_hip_x"],
                     sm[f"{arm}_shoulder_y"] - sm[f"{arm}_hip_y"]).median()
    if not np.isfinite(torso) or torso <= 0:
        return None
    h = ((sm[f"{arm}_shoulder_y"] - sm[f"{arm}_wrist_y"]) / torso).dropna()
    return h.rolling(5, center=True, min_periods=1).mean() if len(h) >= 8 else None


def set_point_from_wrist(sm: pd.DataFrame, arm: str):
    """
    Set point = fin de la PRIMERA subida significativa de la muneca (el balon llega arriba, antes del
    empuje final). Se toma la primera subida cuya velocidad llega a la mitad de la maxima, y no la mas
    rapida: asi, si el empuje final es mas veloz que la subida inicial, no se confunde con ella.
    Usa solo la altura de la muneca, asi que sirve tanto de lado como de frente. En tiros de un solo
    movimiento (sin pausa) el set point coincide con el release.
    """
    h = wrist_height(sm, arm)
    if h is None:
        return None
    fr = h.index.values.astype(float)
    v = np.gradient(h.values, fr)
    vmax = v.max()
    if vmax <= 0:
        return None
    above = np.where(v >= 0.5 * vmax)[0]
    i1 = int(above[0])
    j = i1
    while j + 1 < len(v) and v[j + 1] >= 0.5 * vmax:   # primer tramo continuo de subida rapida
        j += 1
    ip = i1 + int(np.argmax(v[i1:j + 1]))
    stop = np.where(v[ip:] <= 0.30 * v[ip])[0]          # la muneca se frena arriba
    return int(fr[ip + (int(stop[0]) if len(stop) else len(v) - 1 - ip)])


def release_from_wrist(sm: pd.DataFrame, arm: str, frac: float = 0.90, use_elbow: bool = True):
    """
    El tiro sube la muneca en DOS etapas: primero lleva el balon al set point (subida
    rapida, luego una pausa) y despues empuja hacia arriba (subida mas lenta) hasta soltarlo.
    Release = primer frame, DESPUES del set point, en el que la altura de la muneca llega al `frac`
    de su recorrido total (0 = punto mas bajo, 1 = mas alto).
    use_elbow=True ubica el set point con el momento de mayor flexion del codo (validado de lado);
    use_elbow=False lo ubica solo con la muneca (recomendado de frente, donde el angulo del codo
    se ve muy acortado).
    """
    h = wrist_height(sm, arm)
    if h is None:
        return None
    hn = (h - h.min()) / (h.max() - h.min() + 1e-9)
    f_flex = None
    if use_elbow:
        elbow = sm.loc[hn.index, f"{arm}_elbow_angle"].dropna()
        if len(elbow):
            before = elbow[elbow.index < elbow.idxmax()]
            f_flex = before.idxmin() if len(before) else elbow.index[0]
    if f_flex is None:
        f_flex = set_point_from_wrist(sm, arm)
        if f_flex is None:
            f_flex = hn.index[0]
    cross = hn[(hn.index >= f_flex) & (hn >= frac)]
    return int(cross.index[0]) if len(cross) else int(hn.idxmax())


def release_from_elbow(elbow):
    """
    Metodo anterior (solo codo): primer frame donde el codo llega casi a su extension
    maxima despues de su punto mas flexionado. Se conserva para comparar.
    `elbow` es una serie (indice = numero de frame).
    """
    if elbow is None or len(elbow) == 0:
        return None
    mx, f_max = elbow.max(), elbow.idxmax()
    before = elbow[elbow.index < f_max]
    f_min = before.idxmin() if len(before) else elbow.index[0]
    cand = elbow[(elbow.index >= f_min) & (elbow >= mx - 2)]
    return int(cand.index[0]) if len(cand) else int(f_max)


def video_features(df: pd.DataFrame, arm: str = "auto", method: str = "wrist", override=None, offset: int = 0, wrist_frac: float = 0.90):
    """
    Resume un video en un dict de metricas. Se usan percentiles 5/95 en vez
    de min/max estrictos para que un frame con ruido no arruine el resultado.
    Las metricas son del BRAZO QUE TIRA, asi da igual si la camara lo ve por
    la izquierda o por la derecha.
    method: 'wrist' (default) o 'elbow'. offset: frames que se suman al release automatico.
    override: frame de release puesto a mano (gana sobre los metodos automaticos).
    """
    det, smooth = _smoothed(df)
    out = {"detection_rate": 100 * len(det) / max(len(df), 1)}
    if len(det) < MIN_DETECTED_FRAMES:
        return None

    if arm == "auto":
        rng = {s_: smooth[f"{s_}_elbow_angle"].quantile(.95) - smooth[f"{s_}_elbow_angle"].quantile(.05)
               for s_ in ("left", "right")}
        arm = max(rng, key=rng.get)
    out["shooting_arm"] = arm

    rel_w = release_from_wrist(smooth, arm, wrist_frac)
    out["last_frame"] = int(det["frame"].max())
    rel_e = release_from_elbow(smooth[f"{arm}_elbow_angle"].dropna())
    out["release_wrist"], out["release_elbow"] = rel_w, rel_e
    if method == "wrist" and rel_w is not None:
        rel_auto, src = rel_w, "wrist"
    else:
        rel_auto, src = rel_e, "elbow"
    if rel_auto is not None:  # corrige el sesgo medido (frames) y mantiene el frame dentro del video
        rel_auto = int(np.clip(int(rel_auto) + int(offset), smooth.index.min(), smooth.index.max()))
    out["release_auto"] = rel_auto
    if override is not None:
        rel, src = int(override), "manual"
    else:
        rel = rel_auto
    out["release_frame"], out["release_source"] = rel, src

    row_rel = None
    if rel is not None:
        row_rel = smooth.iloc[smooth.index.get_indexer([rel], method="nearest")[0]]
    for joint in JOINTS:
        out[f"{joint}_rel"] = row_rel[f"{arm}_{joint}_angle"] if row_rel is not None else np.nan

    for joint in JOINTS:
        s_ = smooth[f"{arm}_{joint}_angle"].dropna()
        p5, p95 = s_.quantile(.05), s_.quantile(.95)
        out[f"{joint}_min"] = p5
        out[f"{joint}_max"] = p95
        out[f"{joint}_mean"] = s_.mean()
        out[f"{joint}_range"] = p95 - p5
    return out


def load_manual(path: Path):
    """Lee release_manual.csv (columnas: video,frame). Devuelve {nombre_sin_extension: frame}."""
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    m.columns = [c.strip().lower() for c in m.columns]
    if not {"video", "frame"} <= set(m.columns):
        print(f"[!] {path} debe tener columnas 'video' y 'frame'; se ignora.")
        return {}
    return {Path(str(r["video"])).stem.lower(): int(r["frame"]) for _, r in m.iterrows() if pd.notna(r["frame"])}


FEATURES = ([f"{j}_{k}" for j in JOINTS for k in ("min", "max", "mean", "range")]
            + [f"{j}_rel" for j in JOINTS])


# --------------------------------------------------------------------------
# 4. Estadistica simple (solo numpy)
# --------------------------------------------------------------------------
def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / max(len(a) + len(b) - 2, 1))
    return (a.mean() - b.mean()) / (sp + 1e-9)


def auc(pos, neg):
    """Probabilidad de que un valor de 'pos' sea mayor que uno de 'neg' (0.5 = azar)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def best_threshold(good, bad):
    """Umbral unico que mejor separa 'good' de 'bad'. Devuelve (umbral, direccion, accuracy)."""
    vals = np.sort(np.unique(np.concatenate([good, bad])))
    cands = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
    best = (None, None, 0.0)
    for t in cands:
        for direction in (">", "<"):
            pred_good_g = good > t if direction == ">" else good < t
            pred_good_b = bad > t if direction == ">" else bad < t
            acc = (pred_good_g.sum() + (~pred_good_b).sum()) / (len(good) + len(bad))
            if acc > best[2]:
                best = (float(t), direction, float(acc))
    return best


def f_ratio(X, y):
    """Razon entre-clases / dentro-de-clases por feature (para elegir features)."""
    classes = np.unique(y)
    grand = X.mean(0)
    between = sum((y == c).sum() * (X[y == c].mean(0) - grand) ** 2 for c in classes)
    within = sum(((X[y == c] - X[y == c].mean(0)) ** 2).sum(0) for c in classes)
    return between / (within + 1e-9)


def fit_predict(Xtr, ytr, Xte, k=TOP_K_FEATURES):
    """Clasificador de centroide mas cercano sobre las k mejores features (z-score)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    top = np.argsort(-f_ratio(Ztr, ytr))[:k]
    classes = np.unique(ytr)
    cents = np.array([Ztr[ytr == c][:, top].mean(0) for c in classes])
    d = ((Zte[:, top][:, None, :] - cents[None, :, :]) ** 2).sum(2)
    return classes[d.argmin(1)]


def loo_predictions(X, y):
    preds = np.empty(len(y), dtype=object)
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        preds[i] = fit_predict(X[m], y[m], X[i:i + 1])[0]
    return preds


def summarize_preds(y_true, y_pred):
    acc3 = float((y_true == y_pred).mean())
    acc2 = float(((y_true == "bueno") == (y_pred == "bueno")).mean())
    labels = ["bueno", "codo", "brazo"]
    cm = pd.crosstab(pd.Series(y_true, name="real"), pd.Series(y_pred, name="predicho"))
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    return acc3, acc2, cm


# --------------------------------------------------------------------------
# 5. Reporte
# --------------------------------------------------------------------------
def baseline_line(y):
    """Acierto de un 'modelo' tonto que siempre responde 'malo' (referencia contra el azar)."""
    return f"- Referencia (decir siempre 'malo'): {100 * np.mean(np.asarray(y) != 'bueno'):.0f}%"


def errors_section(df, preds, source, pred_log):
    """Registra cada prediccion y devuelve la tabla de videos mal clasificados."""
    wrong = []
    for (_, r), p in zip(df.iterrows(), preds):
        pred_log.append({"prueba": source, "camara": r["camera"], "video": r["video"],
                         "real": r["label"], "predicho": p,
                         "acierto": r["label"] == p, "ruta": r["path"]})
        if r["label"] != p:
            wrong.append((r, p))
    if not wrong:
        return "Videos mal clasificados: ninguno.\n"
    out = ["Videos mal clasificados:\n",
           "| video | real | predicho | tipo de error |", "|:--|:--|:--|:--|"]
    for r, p in wrong:
        kind = ("bueno <-> malo" if (r["label"] == "bueno") != (p == "bueno")
                else "confundio el tipo de error")
        out.append(f"| {r['video']} | {r['label']} | {p} | {kind} |")
    return "\n".join(out) + "\n"


def analyze_camera(cam_df: pd.DataFrame, camera: str, out_dir: Path, pred_log: list):
    lines = [f"## Camara: {camera}  ({len(cam_df)} videos)\n"]
    counts = cam_df["label"].value_counts().to_dict()
    lines.append(f"Videos por clase: {counts}\n")
    lines.append(f"Deteccion de persona (promedio): {cam_df['detection_rate'].mean():.1f}%  "
                 f"(minimo {cam_df['detection_rate'].min():.1f}%)\n")

    good = cam_df[cam_df["label"] == "bueno"]
    if len(good) < 2 or cam_df["label"].nunique() < 2:
        lines.append("_No hay suficientes videos para comparar._\n")
        return "\n".join(lines)

    # --- Separacion por metrica y por tipo de error
    rows = []
    for f in FEATURES:
        r = {"metrica": f, "bueno (media±sd)": f"{good[f].mean():.1f}±{good[f].std():.1f}"}
        aucs = []
        for err in ("codo", "brazo"):
            bad = cam_df[cam_df["label"] == err]
            if len(bad) < 2:
                r[f"{err} (media±sd)"], r[f"AUC vs {err}"] = "-", np.nan
                continue
            a = auc(bad[f], good[f])
            r[f"{err} (media±sd)"] = f"{bad[f].mean():.1f}±{bad[f].std():.1f}"
            r[f"AUC vs {err}"] = max(a, 1 - a)
            aucs.append(max(a, 1 - a))
        r["_score"] = np.mean(aucs) if aucs else 0
        rows.append(r)
    table = pd.DataFrame(rows).sort_values("_score", ascending=False)
    lines.append("### Metricas que mejor separan bueno de malo\n")
    lines.append("AUC: 0.5 = no separa nada (azar), 1.0 = separa perfecto.\n")
    lines.append(table.drop(columns="_score").head(8).round(2).to_markdown(index=False) + "\n")

    # --- Reglas candidatas por tipo de error
    lines.append("### Reglas candidatas (una sola metrica + umbral)\n")
    for err in ("codo", "brazo"):
        bad = cam_df[cam_df["label"] == err]
        if len(bad) < 2:
            continue
        best = None
        for f in FEATURES:
            t, d, acc = best_threshold(good[f].values, bad[f].values)
            if best is None or acc > best[3]:
                best = (f, t, d, acc)
        f, t, d, acc = best
        lines.append(f"- **Error de {err}**: `{f}` {d} {t:.1f}° = tiro bueno "
                     f"(acierta {acc * 100:.0f}% de {len(good) + len(bad)} videos)")
    lines.append("")

    # --- Clasificador leave-one-out
    X = cam_df[FEATURES].values.astype(float)
    y = cam_df["label"].values
    preds = loo_predictions(X, y)
    acc3, acc2, cm = summarize_preds(y, preds)
    lines.append("### Clasificador simple (leave-one-out)\n")
    lines.append(f"- Acierto bueno / codo / brazo: **{acc3 * 100:.0f}%**")
    lines.append(f"- Acierto bueno vs malo: **{acc2 * 100:.0f}%**")
    lines.append(baseline_line(y) + "\n")
    lines.append("Matriz de confusion (filas = real, columnas = predicho):\n")
    lines.append(cm.to_markdown() + "\n")
    lines.append(errors_section(cam_df, preds, f"LOO {camera}", pred_log))

    make_boxplot(cam_df, table["metrica"].head(4).tolist(), camera, out_dir)
    lines.append(f"Grafico: `boxplot_{camera}.png`\n")
    return "\n".join(lines)


def make_boxplot(cam_df, feats, camera, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [c for c in ("bueno", "codo", "brazo") if c in set(cam_df["label"])]
    fig, axes = plt.subplots(1, len(feats), figsize=(4 * len(feats), 4))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, feats):
        data = [cam_df[cam_df["label"] == c][f].values for c in order]
        try:
            ax.boxplot(data, tick_labels=order)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot(data, labels=order)
        for i, d in enumerate(data, start=1):
            ax.scatter(np.random.normal(i, 0.05, len(d)), d, alpha=.6, s=18)
        ax.set_title(f)
        ax.set_ylabel("grados")
    fig.suptitle(f"Camara: {camera}")
    fig.tight_layout()
    fig.savefig(out_dir / f"boxplot_{camera}.png", dpi=120)
    plt.close(fig)


def transfer_report(df: pd.DataFrame, pred_log: list):
    """Entrena con tripode, evalua con las otras camaras."""
    if "tripode" not in set(df["camera"]):
        return ""
    train = df[df["camera"] == "tripode"]
    lines = ["## Robustez: entrenar con tripode, probar con otras camaras\n",
             "Si el acierto cae mucho, las metricas dependen de la perspectiva de la camara.\n"]
    for cam in ("handheld", "piso"):
        test = df[df["camera"] == cam]
        if test.empty:
            continue
        pred = fit_predict(train[FEATURES].values.astype(float), train["label"].values,
                           test[FEATURES].values.astype(float))
        acc3, acc2, cm = summarize_preds(test["label"].values, pred)
        lines.append(f"### tripode -> {cam}\n")
        lines.append(f"- Acierto bueno / codo / brazo: **{acc3 * 100:.0f}%**")
        lines.append(f"- Acierto bueno vs malo: **{acc2 * 100:.0f}%**")
        lines.append(baseline_line(test["label"].values) + "\n")
        lines.append(cm.to_markdown() + "\n")
        lines.append(errors_section(test.reset_index(drop=True), pred, f"tripode->{cam}", pred_log))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 6. Imagenes del frame del release
# --------------------------------------------------------------------------
ES_NAMES = {"elbow": "codo", "shoulder": "hombro", "knee": "rodilla", "hip": "cadera"}
CHAINS = [("shoulder", "elbow"), ("elbow", "wrist"), ("shoulder", "hip"),
          ("hip", "knee"), ("knee", "ankle")]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, org, scale=0.6, fg=(0, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 5), (x + tw + 6, y + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 2), FONT, scale, fg, 2, cv2.LINE_AA)


def save_release_image(video_path: Path, frame_no: int, arm: str, caption: str,
                       out_path: Path, complexity: int, warmup: int = 10, target_h: int = 520):
    """Guarda el frame del release recortado alrededor de la persona, con el
    esqueleto y los angulos del brazo que tira dibujados. Devuelve out_path o None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened() or frame_no is None:
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    extractor = PoseExtractor(model_complexity=complexity)
    start, idx, frame, res = max(0, frame_no - warmup), 0, None, None
    while idx <= frame_no:
        ok, img = cap.read()
        if not ok:
            break
        if idx >= start:  # unos frames de calentamiento para que el tracking se estabilice
            res = extractor.process_frame(img)
            frame = img
        idx += 1
    cap.release()
    extractor.close()
    if frame is None:
        return None

    if res is not None and res.pose_landmarks:
        pts = landmarks_to_xy(res.pose_landmarks.landmark, w, h)
        angles = compute_all_angles(pts)
        xs = np.array([p[0] for p in pts.values()])
        ys = np.array([p[1] for p in pts.values()])
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        x0 = int(max(0, xs.min() - 0.45 * bw)); x1 = int(min(w, xs.max() + 0.45 * bw))
        y0 = int(max(0, ys.min() - 0.40 * bh)); y1 = int(min(h, ys.max() + 0.12 * bh))
        crop = frame[y0:y1, x0:x1]
        sc = target_h / max(crop.shape[0], 1)
        crop = cv2.resize(crop, None, fx=sc, fy=sc)
        P = {k: (v - np.array([x0, y0])) * sc for k, v in pts.items()}

        for side in ("left", "right"):
            hot = side == arm
            for a, b in CHAINS:
                cv2.line(crop, tuple(P[f"{side}_{a}"].astype(int)), tuple(P[f"{side}_{b}"].astype(int)),
                         (0, 220, 0) if hot else (200, 200, 200), 4 if hot else 2, cv2.LINE_AA)
        for j in JOINTS:
            v = angles.get(f"{arm}_{j}_angle")
            if v is not None:
                _label(crop, f"{ES_NAMES[j]} {int(v)}", P[f"{arm}_{j}"] + np.array([8, 0]))
    else:
        crop = cv2.resize(frame, None, fx=target_h / h, fy=target_h / h)
        _label(crop, "sin deteccion", (10, 60), fg=(0, 0, 255))

    if crop.shape[1] < 300:  # recortes muy angostos: rellenar para que quepa el titulo
        crop = cv2.copyMakeBorder(crop, 0, 0, 0, 300 - crop.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    bar = np.zeros((34, crop.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, caption, (8, 23), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.vstack([bar, crop])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    return out_path


def make_contact_sheet(entries: list, camera: str, out_dir: Path, cell_h: int = 500):
    """Una hoja por camara: una columna por clase (bueno / codo / brazo), una fila por toma."""
    order = [c for c in ("bueno", "codo", "brazo") if any(e["label"] == c for e in entries)]
    if not order:
        return None
    cols = {c: [e["path"] for e in entries if e["label"] == c] for c in order}
    n_rows = max(len(v) for v in cols.values())

    def fit(path):
        img = cv2.imread(str(path))
        return cv2.resize(img, None, fx=cell_h / img.shape[0], fy=cell_h / img.shape[0])

    columns = []
    for c in order:
        tiles = [fit(p) for p in cols[c]]
        cw = max(t.shape[1] for t in tiles)
        cells = []
        for r in range(n_rows):
            cell = np.full((cell_h, cw, 3), 40, dtype=np.uint8)
            if r < len(tiles):
                cell[:, :tiles[r].shape[1]] = tiles[r]
            cells.append(cell)
        head = np.zeros((44, cw, 3), dtype=np.uint8)
        cv2.putText(head, c.upper(), (10, 32), FONT, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        columns.append(np.vstack([head] + cells))
    sep = np.full((columns[0].shape[0], 8, 3), 90, dtype=np.uint8)
    row = []
    for i, col in enumerate(columns):
        row += [col, sep] if i < len(columns) - 1 else [col]
    sheet = np.hstack(row)
    path = out_dir / f"hoja_{camera}.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


# --------------------------------------------------------------------------
# 7. Diagnostico del release
# --------------------------------------------------------------------------
def release_diagnostics(df: pd.DataFrame, manual: dict, debug_files: list):
    """Seccion del reporte: desacuerdo entre metodos y (si hay frames manuales) error real."""
    lines = ["## Diagnostico del release\n"]
    if debug_files:
        lines.append("Graficos por camara (azul = altura de muneca, naranja = angulo del codo; "
                     "verde = release por muneca, rojo = por codo, magenta = manual, "
                     "el elegido es el que se usa en las metricas *_rel):\n")
        lines += [f"- `{f}`" for f in debug_files]
        lines.append("")
    both = df.dropna(subset=["release_wrist", "release_elbow"])
    if len(both):
        diff = (both["release_wrist"] - both["release_elbow"]).abs()
        far = both[diff > 5]
        lines.append(f"Videos donde muneca y codo difieren en mas de 5 frames: **{len(far)} de {len(both)}**")
        if len(far):
            lines.append("(los mas sospechosos, revisalos primero): " + ", ".join(far["video"].tolist()))
        lines.append("")

    if "last_frame" in df.columns:
        edge = df[(df["release_auto"] >= df["last_frame"] - 3) | (df["release_auto"] <= 3)]
        if len(edge):
            lines.append(f"Videos donde el release automatico cae en los primeros/ultimos 3 frames "
                         f"(casi seguro mal, el clip deberia tener movimiento despues del release): "
                         f"**{len(edge)}** -> " + ", ".join(edge["video"].tolist()) + "\n")
    rows = []
    for _, r in df.iterrows():
        k = Path(r["video"]).stem.lower()
        if k in manual:
            rows.append({"video": r["video"], "manual": manual[k], "muneca": r["release_wrist"],
                         "codo": r["release_elbow"], "auto": r.get("release_auto", np.nan)})
    if rows:
        t = pd.DataFrame(rows)
        lines.append("### Error contra tus frames manuales\n")
        for m, name in (("muneca", "muneca"), ("codo", "codo"), ("auto", "elegido (metodo + offset)")):
            err = (t[m] - t["manual"]).abs().dropna()
            if len(err):
                lines.append(f"- Metodo **{name}**: error mediano {err.median():.1f} frames; "
                             f"dentro de ±3 frames: {100 * (err <= 3).mean():.0f}% de {len(err)} videos")
        for m in ("muneca", "codo"):
            bias = (t[m] - t["manual"]).median()
            lines.append(f"- Sesgo mediano del metodo {m}: **{bias:+.1f} frames** "
                         f"(negativo = detecta antes que tu; corregirlo: `--release-offset {int(round(-bias))}`)")
        t["error muneca (+ = tarde)"] = t["muneca"] - t["manual"]
        t["error codo (+ = tarde)"] = t["codo"] - t["manual"]
        t = t.drop(columns=["auto"])
        lines.append("")
        lines.append(t.round(0).to_markdown(index=False))
        lines.append("")
    else:
        lines.append("_Aun no hay frames manuales. Crea `release_manual.csv` (columnas: video,frame) "
                     "en la carpeta de resultados para medir el error real._\n")
    return "\n".join(lines)


def wrist_sweep_section(store: dict, manual: dict, fracs=(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)):
    """Como cambia el error del metodo de muneca segun el umbral, contra tus frames manuales."""
    rows = []
    for frac in fracs:
        errs = []
        for stem, m in manual.items():
            if stem in store:
                _, angles, f = store[stem]
                r = release_from_wrist(_smoothed(angles)[1], f["shooting_arm"], frac)
                if r is not None:
                    errs.append(r - m)
        if errs:
            e = np.array(errs, float)
            rows.append({"umbral": frac, "error absoluto mediano": np.median(np.abs(e)),
                         "dentro de ±3 frames": f"{100 * (np.abs(e) <= 3).mean():.0f}%",
                         "sesgo mediano (+ = tarde)": np.median(e), "videos": len(e)})
    if not rows:
        return ""
    return ("### Metodo muneca: efecto del umbral (--wrist-frac)\n\n"
            + pd.DataFrame(rows).to_markdown(index=False) + "\n")


def make_release_debug(entries: list, camera: str, out_dir: Path):
    """Una grafica por toma: altura de muneca y angulo del codo con el release elegido marcado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [c for c in ("bueno", "codo", "brazo") if any(e["label"] == c for e in entries)]
    if not order:
        return None
    cols = {c: [e for e in entries if e["label"] == c] for c in order}
    n_rows = max(len(v) for v in cols.values())
    fig, axes = plt.subplots(n_rows, len(order), figsize=(4.4 * len(order), 1.9 * n_rows + 0.6), squeeze=False)

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    for j, c in enumerate(order):
        for i in range(n_rows):
            ax = axes[i][j]
            if i >= len(cols[c]):
                ax.axis("off")
                continue
            e = cols[c][i]
            if e["h"] is not None and len(e["h"]) > 1:
                ax.plot(e["h"].index, norm(e["h"]), color="tab:blue", lw=1.2)
            if len(e["elbow_series"]) > 1:
                ax.plot(e["elbow_series"].index, norm(e["elbow_series"]), color="tab:orange", lw=1.2)
            for key, color, ls in (("f_wrist", "green", "--"), ("f_elbow", "red", ":"), ("f_manual", "magenta", "-")):
                if e[key] is not None:
                    ax.axvline(e[key], color=color, ls=ls, lw=1.4)
            ax.set_title(f"{c} | {e['stem']}", fontsize=8)
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Release por toma - camara {camera}", fontsize=10)
    fig.tight_layout()
    path = out_dir / f"debug_release_{camera}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def save_filmstrip(video_path: Path, df: pd.DataFrame, arm: str, out_path: Path, marks: list,
                   frame_range=None, max_tiles: int = 40, tile_h: int = 220):
    """
    Tira de frames recortados con numero de frame, para encontrar a ojo el release real.
    marks: lista de (frame, color BGR) que se enmarcan en la tira.
    """
    det, sm = _smoothed(df)
    xcols, ycols = [f"{n}_x" for n in LANDMARK_IDS], [f"{n}_y" for n in LANDMARK_IDS]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened() or any(c not in sm.columns for c in xcols + ycols):
        return None
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    xmin, xmax = np.nanpercentile(sm[xcols].values, [1, 99])
    ymin, ymax = np.nanpercentile(sm[ycols].values, [1, 99])
    bw, bh = xmax - xmin, ymax - ymin
    x0, x1 = int(max(0, xmin - 0.40 * bw)), int(min(w, xmax + 0.40 * bw))
    y0, y1 = int(max(0, ymin - 0.35 * bh)), int(min(h, ymax + 0.10 * bh))

    lo, hi = frame_range if frame_range else (int(det["frame"].min()), int(det["frame"].max()))
    step = max(1, int(np.ceil((hi - lo + 1) / max_tiles)))
    frames = sorted(set(range(lo, hi + 1, step)) | {int(f) for f, _ in marks if f is not None and lo <= f <= hi})
    borders = {}
    for f, color in marks:
        if f is not None:
            borders.setdefault(int(f), []).append(color)

    tiles, idx, wanted = [], 0, set(frames)
    while idx <= frames[-1]:
        ok, img = cap.read()
        if not ok:
            break
        if idx in wanted:
            crop = img[y0:y1, x0:x1]
            sc = tile_h / max(crop.shape[0], 1)
            tile = cv2.resize(crop, None, fx=sc, fy=sc)
            if idx in sm.index:
                r = sm.loc[idx]
                P = {n: (np.array([r[f"{n}_x"] - x0, r[f"{n}_y"] - y0]) * sc) for n in LANDMARK_IDS}
                if all(np.isfinite(p).all() for p in P.values()):
                    for side in ("left", "right"):
                        for a, b in CHAINS:
                            cv2.line(tile, tuple(P[f"{side}_{a}"].astype(int)), tuple(P[f"{side}_{b}"].astype(int)),
                                     (0, 220, 0) if side == arm else (200, 200, 200), 3 if side == arm else 1, cv2.LINE_AA)
            _label(tile, str(idx), (4, 24), scale=0.7)
            for k, color in enumerate(borders.get(idx, [])):
                cv2.rectangle(tile, (2 * k, 2 * k), (tile.shape[1] - 1 - 2 * k, tile.shape[0] - 1 - 2 * k), color, 3)
            tiles.append(tile)
        idx += 1
    cap.release()
    if not tiles:
        return None

    tw = tiles[0].shape[1]
    ncols = max(3, min(8, 1800 // max(tw, 1)))
    while len(tiles) % ncols:
        tiles.append(np.full_like(tiles[0], 40))
    grid = np.vstack([np.hstack(tiles[i:i + ncols]) for i in range(0, len(tiles), ncols)])
    bar = np.zeros((34, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"{video_path.stem} | paso {step} frame(s) | verde=release muneca  rojo=release codo  "
                     f"blanco=elegido  magenta=manual", (8, 23), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([bar, grid]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Compara videos de tiro libre buenos vs malos")
    ap.add_argument("--videos", required=True, help="Carpeta raiz con los videos")
    ap.add_argument("--output", default="experimento", help="Carpeta de resultados")
    ap.add_argument("--arm", default="auto", choices=["auto", "left", "right"],
                    help="Brazo con el que tira la persona (auto = lo deduce)")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--force", action="store_true", help="Reprocesar aunque exista cache")
    ap.add_argument("--no-images", action="store_true", help="No generar imagenes del release")
    ap.add_argument("--release-method", default="wrist", choices=["wrist", "elbow"],
                    help="Como detectar el release (wrist = altura de muneca, elbow = extension del codo)")
    ap.add_argument("--wrist-frac", type=float, default=0.90,
                    help="Con --release-method wrist: fraccion (0-1) del recorrido de la muneca que marca el release")
    ap.add_argument("--release-offset", type=int, default=0,
                    help="Frames a sumar al release automatico (corrige un sesgo medido con frames manuales)")
    ap.add_argument("--manual", default=None,
                    help="CSV con frames de release puestos a mano (columnas: video,frame). "
                         "Por defecto busca <salida>/release_manual.csv")
    ap.add_argument("--strips", nargs="+", metavar="VIDEO",
                    help="Solo generar tiras de frames para estos videos (nombre sin extension) y salir")
    ap.add_argument("--strip-range", nargs=2, type=int, metavar=("DESDE", "HASTA"),
                    help="Con --strips: limitar la tira a este rango de frames (para ver frame por frame)")
    args = ap.parse_args()

    root, out_dir = Path(args.videos), Path(args.output)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos, unknown = find_videos(root)
    if not videos:
        sys.exit("No encontre videos reconocibles. Revisa que la ruta o el nombre incluya "
                 "camara (tripode/handheld/piso) y clase (bueno/codo/brazo).")
    if unknown:
        print(f"[!] {len(unknown)} videos ignorados (no se pudo deducir camara/clase):")
        for p in unknown:
            print("    ", p)

    manual = load_manual(Path(args.manual) if args.manual else out_dir / "release_manual.csv")
    if manual:
        print(f"Frames de release manuales cargados: {len(manual)}")
    rows, release_imgs, debug_entries, store = [], [], [], {}
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v['camera']:9s} {v['label']:6s} {v['path'].name}")
        cache = cache_dir / f"{v['camera']}__{v['label']}__{v['path'].stem}.csv"
        angles = extract_angles(v["path"], cache, args.complexity, args.force)
        override = manual.get(v["path"].stem.lower())
        feats = video_features(angles, args.arm, args.release_method, override, args.release_offset, args.wrist_frac)
        if feats is None:
            print("     [!] casi no se detecto persona, se excluye")
            continue
        rows.append({"video": v["path"].name, "camera": v["camera"], "label": v["label"],
                     "path": str(v["path"]), **feats})
        arm_used = feats["shooting_arm"]
        _, sm_ = _smoothed(angles)
        debug_entries.append({"camera": v["camera"], "label": v["label"], "stem": v["path"].stem,
                              "h": wrist_height(sm_, arm_used), "elbow_series": sm_[f"{arm_used}_elbow_angle"].dropna(),
                              "f_wrist": feats["release_wrist"], "f_elbow": feats["release_elbow"],
                              "f_manual": override})
        store[v["path"].stem.lower()] = (v["path"], angles, feats)
        if not args.no_images and not args.strips:
            try:
                cap_txt = f"{v['path'].stem} f{feats['release_frame']}"
                img_path = save_release_image(
                    v["path"], feats["release_frame"], feats["shooting_arm"], cap_txt,
                    out_dir / "release" / v["camera"] / f"{v['label']}__{v['path'].stem}.jpg",
                    args.complexity)
                if img_path:
                    release_imgs.append({"camera": v["camera"], "label": v["label"], "path": img_path})
            except Exception as e:  # una imagen fallida no debe tumbar el experimento
                print(f"     [!] no se pudo generar la imagen del release: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "features.csv", index=False)

    if args.strips:
        colors = {"wrist": (0, 200, 0), "elbow": (0, 0, 255), "chosen": (255, 255, 255), "manual": (255, 0, 255)}
        for name in args.strips:
            key = Path(name).stem.lower()
            if key not in store:
                print(f"[!] No encontre el video '{name}' entre los analizados.")
                continue
            path_, angles_, f_ = store[key]
            marks = [(f_["release_wrist"], colors["wrist"]), (f_["release_elbow"], colors["elbow"]),
                     (f_["release_frame"], colors["chosen"])]
            if key in manual:
                marks.append((manual[key], colors["manual"]))
            outp = save_filmstrip(path_, angles_, f_["shooting_arm"], out_dir / "tiras" / f"{path_.stem}.jpg",
                                  marks, frame_range=tuple(args.strip_range) if args.strip_range else None)
            print("Tira:", outp)
        print("\nAbre las tiras, encuentra el frame donde el balon sale de la mano y anotalo en "
              f"{out_dir / 'release_manual.csv'} (columnas: video,frame).")
        return

    pred_log = []
    report = ["# Resultados del experimento de tiro libre\n"]
    for camera in [c for c in ("tripode", "handheld", "piso") if c in set(df["camera"])]:
        report.append(analyze_camera(df[df["camera"] == camera].reset_index(drop=True),
                                     camera, out_dir, pred_log))
    report.append(transfer_report(df, pred_log))
    sheets = []
    for camera in [c for c in ("tripode", "handheld", "piso") if c in set(df["camera"])]:
        sp = make_contact_sheet([e for e in release_imgs if e["camera"] == camera], camera, out_dir)
        if sp:
            sheets.append(f"- `{sp.name}`")
    debug_files = []
    for camera in [c for c in ("tripode", "handheld", "piso") if c in set(df["camera"])]:
        dp = make_release_debug([e for e in debug_entries if e["camera"] == camera], camera, out_dir)
        if dp:
            debug_files.append(dp.name)
    report.append(release_diagnostics(df, manual, debug_files))
    if manual:
        report.append(wrist_sweep_section(store, manual))
    if sheets:
        report.append("## Imagenes del frame de release\n\n" + "\n".join(sheets) +
                      "\n\nUna hoja por camara, una columna por clase. Verifica que el frame sea el "
                      "release y que el esqueleto siga bien el cuerpo.\n")
    pd.DataFrame(pred_log).to_csv(out_dir / "predicciones.csv", index=False)

    text = "\n".join(report)
    (out_dir / "reporte.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nListo. Resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
