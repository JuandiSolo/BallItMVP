#!/usr/bin/env python3
"""
analyze_elbow.py
----------------
Mide el CODO ABIERTO ANTES DEL TIRO (en la preparacion / set point), no en el release.

Idea: quien tira mal suele llevar el codo hacia afuera (a la derecha, si tira con la derecha)
ANTES de lanzar. De FRENTE eso se ve como una separacion horizontal entre el codo y el hombro.

Metricas (todas en 'anchos de hombro', asi no dependen del zoom ni de la estatura):
  flare           = cuanto se aleja el codo del hombro hacia afuera (0 = codo bajo el hombro)
  elbow_vs_wrist  = cuanto mas afuera esta el codo que la muñeca (antebrazo inclinado hacia adentro)
  tilt            = inclinacion del antebrazo respecto a la vertical (grados)
  abduction       = angulo cadera-hombro-codo (cuanto se abre el brazo respecto al torso)
Cada una se resume en tres momentos: en el set point (_set), maximo antes del set point (_pre_max)
y promedio de la fase de preparacion (_pre_mean).

Modos (se pueden combinar):
  --videos CARPETA        videos etiquetados (bueno / codo / brazo en el nombre o la carpeta)
  --test CARPETA          videos de OTRAS personas, sin etiquetas (usalo junto con --videos)
  --check CARPETA         verifica a ojo set point, release y momento de mayor apertura del codo
  --compare ANTES DESPUES compara dos videos (¿mejoro el codo?)
  --find-shots VIDEO      encuentra cada tiro en un video largo (--cut para recortarlos)

Ejemplos:
  python analyze_elbow.py --videos frente/ --arm right
  python analyze_elbow.py --videos frente/ --test sujetos/ --arm right
  python analyze_elbow.py --find-shots largo.mov --arm right --cut
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import experiment as E
from src.angle_calculator import LANDMARK_IDS

FRONT_METRICS = ["flare", "elbow_vs_wrist", "abduction"]   # tilt se calcula pero no entra a FEATS (se invierte con la muneca baja)
WINDOWS = ("set", "pre_max", "pre_mean", "lift_max", "lift_mean", "prep_max", "prep_mean")
FEATS = [f"{m}_{k}" for m in FRONT_METRICS for k in WINDOWS]
HEADLINE = "flare_lift_max"   # pico de apertura del codo durante todo el levantamiento (metrica por defecto)
RULE_VERSION = 3
RULE_FAMILIES = ("_lift_", "_prep_")   # ventanas que NO dependen de que el tiro tenga pausa
DEFAULT_RULE = {"version": RULE_VERSION, "feature": HEADLINE, "t": 0.38, "direction": "<", "band": 0.04,
                "good_mean": 0.32, "good_sd": 0.06, "bad_mean": 0.48, "bad_sd": 0.10, "n_good": 5, "n_bad": 5,
                "source": "regla de ejemplo (de un reporte anterior); corre --videos para calcular la tuya"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# Metricas frontales
# --------------------------------------------------------------------------
def frontal_series(sm: pd.DataFrame, arm: str):
    """Serie por frame de las metricas frontales del brazo que tira."""
    other = "left" if arm == "right" else "right"
    need = [f"{arm}_shoulder_x", f"{other}_shoulder_x", f"{arm}_elbow_x", f"{arm}_wrist_x",
            f"{arm}_elbow_y", f"{arm}_wrist_y", f"{arm}_shoulder_angle"]
    if any(c not in sm.columns for c in need):
        return None
    sw = sm[f"{arm}_shoulder_x"] - sm[f"{other}_shoulder_x"]
    sign = np.sign(sw.median()) or 1.0          # sentido 'hacia afuera' (no depende de si la imagen esta en espejo)
    width = sw.abs().median()                    # ancho de hombros (constante por video)
    if not np.isfinite(width) or width <= 0:
        return None
    ex, sx, wx = sm[f"{arm}_elbow_x"], sm[f"{arm}_shoulder_x"], sm[f"{arm}_wrist_x"]
    return pd.DataFrame({
        "flare": sign * (ex - sx) / width,
        "elbow_vs_wrist": sign * (ex - wx) / width,
        "tilt": np.degrees(np.arctan2(sign * (ex - wx), sm[f"{arm}_elbow_y"] - sm[f"{arm}_wrist_y"])),
        "abduction": sm[f"{arm}_shoulder_angle"],
    })


def elbow_features(sm: pd.DataFrame, arm: str, frac: float = 0.90, prep_level: float = 0.60):
    """
    Metricas del codo en la fase PREVIA al tiro. Ventanas:
      set        = alrededor del set point (+-2 frames)
      pre_*      = desde que el balon empieza a subir hasta el set point
      lift_*     = desde que el balon empieza a subir hasta el release (NO depende de ubicar bien el set point)
      prep_*     = desde que el balon empieza a subir hasta que la muneca llega al `prep_level` de su recorrido
                   (o hasta el set point, lo que ocurra primero): la preparacion, sin la extension final del brazo.
                   Tampoco depende de que el tiro tenga pausa.
    Devuelve tambien el frame del 'pico': donde el codo estuvo mas abierto durante el levantamiento.
    """
    h = E.wrist_height(sm, arm)
    series = frontal_series(sm, arm)
    if h is None or series is None:
        return None
    set_f = E.set_point_from_wrist(sm, arm)
    rel_f = E.release_from_wrist(sm, arm, frac, use_elbow=False)   # de frente el angulo del codo no sirve
    if set_f is None or rel_f is None:
        return None
    hn = (h - h.min()) / (h.max() - h.min() + 1e-9)
    low = hn[(hn.index <= set_f) & (hn <= 0.10)]
    start_f = int(low.index[-1]) if len(low) else int(hn.index[0])  # inicio del levantamiento del balon
    end_f = max(set_f, rel_f)
    lvl = hn[(hn.index >= start_f) & (hn >= prep_level)]
    prep_end = min(set_f, int(lvl.index[0])) if len(lvl) else set_f

    def win(a, b):
        return series[(series.index >= a) & (series.index <= b)]

    pre, lift, around = win(start_f, set_f), win(start_f, end_f), win(set_f - 2, set_f + 2)
    prep = win(start_f, prep_end)
    if len(prep) < 2:
        prep = pre
    if len(pre) < 3 or len(lift) < 3 or len(around) == 0 or lift["flare"].dropna().empty:
        return None
    out = {"has_pause": bool(rel_f - set_f >= 3), "gap_frames": int(rel_f - set_f),
           "start_frame": int(start_f), "set_frame": int(set_f), "release_frame": int(rel_f),
           "prep_frame": int(prep_end), "peak_frame": int(lift["flare"].idxmax()),
           "tilt_set": float(around["tilt"].median())}
    for m in FRONT_METRICS:
        out[f"{m}_set"] = float(around[m].median())
        out[f"{m}_pre_max"], out[f"{m}_pre_mean"] = float(pre[m].max()), float(pre[m].mean())
        out[f"{m}_lift_max"], out[f"{m}_lift_mean"] = float(lift[m].max()), float(lift[m].mean())
        out[f"{m}_prep_max"], out[f"{m}_prep_mean"] = float(prep[m].max()), float(prep[m].mean())
    return out


def process_video(path: Path, out_dir: Path, args):
    cache = out_dir / "cache" / f"{path.parent.name}__{path.stem}.csv"
    df = E.extract_angles(path, cache, args.complexity, args.force)
    det, sm = E._smoothed(df)
    if len(det) < E.MIN_DETECTED_FRAMES:
        return None
    f = elbow_features(sm, args.arm, args.frac, args.prep_level)
    if f is None:
        return None
    f["detection_rate"] = 100 * len(det) / max(len(df), 1)
    return df, sm, f


# --------------------------------------------------------------------------
# Imagenes
# --------------------------------------------------------------------------
def grab_frames(video_path: Path, frames):
    want = sorted({int(f) for f in frames if f is not None})
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened() or not want:
        return {}, (0, 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wanted, out, idx = set(want), {}, 0
    while idx <= want[-1]:
        ok, img = cap.read()
        if not ok:
            break
        if idx in wanted:
            out[idx] = img
        idx += 1
    cap.release()
    return out, (w, h)


def person_bbox(sm, w, h, lo=None, hi=None):
    d = sm if lo is None else sm[(sm.index >= lo) & (sm.index <= hi)]
    xs = d[[f"{n}_x" for n in LANDMARK_IDS]].values
    ys = d[[f"{n}_y" for n in LANDMARK_IDS]].values
    if not np.isfinite(xs).any() or not np.isfinite(ys).any():
        return None
    xmin, xmax = np.nanpercentile(xs, [1, 99])
    ymin, ymax = np.nanpercentile(ys, [1, 99])
    bw, bh = xmax - xmin, ymax - ymin
    return (int(max(0, xmin - 0.45 * bw)), int(max(0, ymin - 0.40 * bh)),
            int(min(w, xmax + 0.45 * bw)), int(min(h, ymax + 0.12 * bh)))


def make_tile(img, sm, frame, arm, bbox, caption, note="", target_h=460):
    x0, y0, x1, y1 = bbox
    crop = img[y0:y1, x0:x1].copy()
    sc = target_h / max(crop.shape[0], 1)
    crop = cv2.resize(crop, None, fx=sc, fy=sc)
    r = sm.iloc[sm.index.get_indexer([frame], method="nearest")[0]]
    P = {n: np.array([r[f"{n}_x"] - x0, r[f"{n}_y"] - y0]) * sc for n in LANDMARK_IDS}
    if all(np.isfinite(p).all() for p in P.values()):
        for side in ("left", "right"):
            for a, b in E.CHAINS:
                cv2.line(crop, tuple(P[f"{side}_{a}"].astype(int)), tuple(P[f"{side}_{b}"].astype(int)),
                         (0, 220, 0) if side == arm else (200, 200, 200), 4 if side == arm else 2, cv2.LINE_AA)
        x = int(P[f"{arm}_shoulder"][0])  # linea vertical por el hombro: el codo deberia quedar cerca de ella
        cv2.line(crop, (x, 0), (x, crop.shape[0]), (255, 255, 0), 1, cv2.LINE_AA)
    if note:
        E._label(crop, note, (6, 52))
    if crop.shape[1] < 300:
        crop = cv2.copyMakeBorder(crop, 0, 0, 0, 300 - crop.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
    bar = np.zeros((34, crop.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, caption, (8, 23), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, crop])


def frame_tile(path, sm, f, which, caption, arm, preloaded=None):
    """
    Imagen recortada de un momento del tiro: which = 'peak' (codo mas abierto), 'set' o 'release'.
    preloaded = (imgs, (w, h)) de una sola lectura del video (grab_frames) para no releerlo por cada tiro.
    """
    frame = f[f"{which}_frame"]
    imgs, (w, h) = preloaded if preloaded is not None else grab_frames(path, [frame])
    if frame not in imgs:
        return None
    bbox = person_bbox(sm, w, h, f["start_frame"], max(f["set_frame"], f["release_frame"]) + 5)
    if bbox is None:
        return None
    note = {"peak": f"flare max {f['flare_lift_max']:+.2f}", "set": f"set  flare {f['flare_set']:+.2f}",
            "release": "release"}[which]
    return make_tile(imgs[frame], sm, frame, arm, bbox, caption, note=note)


def list_videos(folder: Path):
    """Videos de una carpeta; si no existe o esta vacia, avisa claro y termina."""
    if not folder.is_dir():
        sys.exit(f"\n[X] No existe la carpeta '{folder}'. Revisa el nombre y que la corras desde la raiz del repo "
                 f"({Path.cwd()}).")
    vids = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in E.VIDEO_EXTS]
    if not vids:
        sys.exit(f"\n[X] La carpeta '{folder}' no tiene videos ({', '.join(sorted(E.VIDEO_EXTS))}).")
    return vids


def resolve_video(p) -> Path:
    """Acepta el nombre aunque cambie la extension o las mayusculas; si no existe, lo dice claro."""
    p = Path(p)
    if p.is_file():
        return p
    if p.parent.is_dir():
        alts = [q for q in p.parent.iterdir() if q.stem.lower() == p.stem.lower() and q.suffix.lower() in E.VIDEO_EXTS]
        if alts:
            print(f"[i] No existe {p.name}; uso {alts[0].name}")
            return alts[0]
        names = sorted(q.name for q in p.parent.iterdir() if q.suffix.lower() in E.VIDEO_EXTS)[:8]
        sys.exit(f"\n[X] No existe el video '{p}'. Videos en {p.parent}: {names}")
    sys.exit(f"\n[X] No existe la carpeta '{p.parent}' (video '{p.name}').")


def sheet(columns: dict, out_path: Path, cell_h: int = 460):
    """columns = {nombre: [tiles]} -> una columna por nombre, una fila por tile."""
    columns = {k: v for k, v in columns.items() if v}
    if not columns:
        return None
    n_rows = max(len(v) for v in columns.values())
    cols = []
    for name, tiles in columns.items():
        tiles = [cv2.resize(t, None, fx=cell_h / t.shape[0], fy=cell_h / t.shape[0]) for t in tiles]
        cw = max(t.shape[1] for t in tiles)
        cells = []
        for r in range(n_rows):
            cell = np.full((cell_h, cw, 3), 40, dtype=np.uint8)
            if r < len(tiles):
                cell[:, :tiles[r].shape[1]] = tiles[r]
            cells.append(cell)
        head = np.zeros((44, cw, 3), dtype=np.uint8)
        cv2.putText(head, name.upper(), (10, 32), FONT, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        cols.append(np.vstack([head] + cells))
    sep = np.full((cols[0].shape[0], 8, 3), 90, dtype=np.uint8)
    row = []
    for i, c in enumerate(cols):
        row += [c, sep] if i < len(cols) - 1 else [c]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


# --------------------------------------------------------------------------
# Analisis de videos etiquetados
# --------------------------------------------------------------------------
def find_labeled(root: Path):
    found = []
    for p in list_videos(root):
        label = E._match(E._norm(str(p.relative_to(root))), E.CLASS_KEYS)
        if label:
            found.append((p, label))
        else:
            print(f"[!] ignorado (sin bueno/codo/brazo en el nombre ni en la carpeta): {p}")
    if not found:
        sys.exit(f"\n[X] Hay videos en '{root}' pero ninguno trae 'bueno' o 'codo' en el nombre o en su carpeta "
                 f"(ejemplo: {root}/bueno/video1.mov).")
    return found


def loo_rule_accuracy(df: pd.DataFrame, y: np.ndarray, feature: str) -> float:
    """Aprende el umbral SIN un video y lo predice; se repite para cada video."""
    vals, ok = df[feature].values.astype(float), 0
    for i in range(len(df)):
        keep = np.arange(len(df)) != i
        g, b = vals[keep & (y == "bueno")], vals[keep & (y == "malo")]
        if len(g) == 0 or len(b) == 0:
            continue
        t, d, _ = E.best_threshold(g, b)
        pred = "bueno" if ((vals[i] > t) if d == ">" else (vals[i] < t)) else "malo"
        ok += int(pred == y[i])
    return ok / len(df)


def analyze_labeled(df: pd.DataFrame):
    """Texto del reporte + regla aprendida. df: columnas video,label + FEATS."""
    y = np.where(df["label"] == "bueno", "bueno", "malo")
    good, bad = df[y == "bueno"], df[y == "malo"]
    lines = [f"## Videos etiquetados ({len(df)}): {df['label'].value_counts().to_dict()}\n"]
    if len(good) < 2 or len(bad) < 2:
        lines.append("_Se necesitan al menos 2 videos buenos y 2 malos para comparar._\n")
        return "\n".join(lines), None

    rows = []
    for f in FEATS:
        a = E.auc(bad[f], good[f])
        rows.append({"metrica": f, "bueno (media±sd)": f"{good[f].mean():.2f}±{good[f].std():.2f}",
                     "malo (media±sd)": f"{bad[f].mean():.2f}±{bad[f].std():.2f}",
                     "AUC": max(a, 1 - a), "malo es mayor": bool(a > 0.5)})
    table = pd.DataFrame(rows)
    table["efecto"] = [abs(E.cohen_d(bad[f], good[f])) for f in table["metrica"]]
    table = table.sort_values(["AUC", "efecto"], ascending=False).drop(columns="efecto")
    lines.append("### Que metricas separan bueno de malo\n")
    lines.append("AUC: 0.5 = azar, 1.0 = separa perfecto. Unidades: flare y elbow_vs_wrist en anchos de hombro; "
                 "abduction en grados.\n")
    lines.append(table.head(8).round(2).to_markdown(index=False) + "\n")

    # La regla solo puede usar ventanas que no dependen de acertar el set point (*_lift_*, *_prep_*): asi tambien vale
    # para quien tira en un solo movimiento, sin pausa.
    cands = table[table["metrica"].apply(lambda m: any(k in m for k in RULE_FAMILIES))]
    best = cands.iloc[0]["metrica"]
    t, direction, acc = E.best_threshold(good[best].values, bad[best].values)
    sd_pool = float(np.sqrt((good[best].var(ddof=1) + bad[best].var(ddof=1)) / 2))
    rule = {"version": RULE_VERSION, "feature": best, "t": float(t), "direction": direction,
            "band": float(max(0.5 * sd_pool, 1e-6)),
            "good_mean": float(good[best].mean()), "good_sd": float(good[best].std()),
            "bad_mean": float(bad[best].mean()), "bad_sd": float(bad[best].std()),
            "n_good": int(len(good)), "n_bad": int(len(bad))}
    loo = loo_rule_accuracy(df, y, best)
    lines.append("### Regla (una sola metrica)\n")
    lines.append(f"- `{best}` {direction} {t:.2f} = tiro bueno. Con todos los videos acierta {acc * 100:.0f}%.")
    lines.append(f"- **Leave-one-out** (el umbral se aprende sin el video que se evalua): **{loo * 100:.0f}%** "
                 f"(decir siempre 'malo': {100 * (y == 'malo').mean():.0f}%).")
    lines.append(f"- Zona dudosa: ±{rule['band']:.2f} alrededor del umbral.\n")
    from math import comb
    p_perfect = 2 / comb(len(good) + len(bad), len(good))
    lines.append(f"_Ojo: con {len(good)} vs {len(bad)} videos, una metrica que no sirve de nada saca AUC 1.0 el "
                 f"{p_perfect:.1%} de las veces, y aqui se miraron {len(FEATS)}. Por eso lo que cuenta es que "
                 f"varias formas distintas de medir lo mismo coincidan, y que se repita con otras personas._\n")
    if "has_pause" in df.columns:
        pg, pb = int(good["has_pause"].sum()), int(bad["has_pause"].sum())
        rule["pause_good"], rule["pause_bad"] = pg / len(good), pb / len(bad)
        lines.append(f"Videos CON pausa en el set point: buenos {pg} de {len(good)}, malos {pb} de {len(bad)}. "
                     "_Si los malos tiran mas de corrido, parte de lo que se separa puede ser el ritmo y no el codo; "
                     "por eso la regla usa solo geometria del codo._\n")
    return "\n".join(lines), rule


def verdict(value, rule) -> str:
    """'bueno' / 'dudoso' / 'malo' segun la regla. Dudoso = a menos de media desviacion del umbral."""
    if value is None or not np.isfinite(value):
        return "?"
    if abs(value - rule["t"]) < rule.get("band", 0.0):
        return "dudoso"
    good_side = (value > rule["t"]) if rule["direction"] == ">" else (value < rule["t"])
    return "bueno" if good_side else "malo"


def save_rule(out_dir: Path, rule: dict, args) -> None:
    data = {**rule, "arm": args.arm, "frac": args.frac, "prep_level": args.prep_level}
    (out_dir / "rule.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_rule(out_dir: Path):
    p = out_dir / "rule.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return d if d.get("version") == RULE_VERSION else None


def apply_rule(value, rule):
    return "bueno" if ((value > rule["t"]) if rule["direction"] == ">" else (value < rule["t"])) else "malo"


# --------------------------------------------------------------------------
# Tiros en un video largo
# --------------------------------------------------------------------------
def find_shots(sm: pd.DataFrame, arm: str, fps: float, frac: float, use_elbow: bool,
               min_height: float = 0.5, min_rise: float = 0.5, min_gap: int = 0):
    """
    Encuentra cada tiro: la muñeca sube por encima del hombro (>= min_height largos de torso) y venia de
    estar al menos min_rise mas abajo en los 2 s previos. Cada tiro devuelve ventana, set point y release.
    """
    h = E.wrist_height(sm, arm)
    if h is None:
        return []
    fr = np.arange(int(h.index.min()), int(h.index.max()) + 1)
    vals = h.reindex(fr).interpolate(limit=int(fps * 0.5), limit_direction="both").values
    W, back = int(round(0.7 * fps)), int(round(2 * fps))
    peaks = []
    for i in range(len(vals)):
        if not np.isfinite(vals[i]) or vals[i] < min_height:
            continue
        if vals[i] < np.nanmax(vals[max(0, i - W):i + W + 1]):
            continue
        if vals[i] - np.nanmin(vals[max(0, i - back):i + 1]) < min_rise:
            continue
        peaks.append(i)
    merged = []
    for i in peaks:  # picos a menos de 1.5 s son el mismo tiro: se queda el mas alto
        if merged and i - merged[-1] < 1.5 * fps:
            if vals[i] > vals[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)

    shots = []
    for i in merged:
        lo = max(0, i - back)
        trough = lo + int(np.nanargmin(vals[lo:i + 1]))
        a = int(fr[max(0, trough - int(0.3 * fps))])
        b = int(fr[min(len(fr) - 1, i + int(0.7 * fps))])
        win = sm[(sm.index >= a) & (sm.index <= b)]
        if len(win) < 8:
            continue
        set_f = E.set_point_from_wrist(win, arm)
        rel_f = E.release_from_wrist(win, arm, frac, use_elbow=use_elbow)
        if set_f is None or rel_f is None or rel_f - set_f < min_gap:
            continue  # un tiro real tiene set point y release separados (subida en dos etapas)
        shots.append({"start": a, "end": b, "peak": int(fr[i]), "set": set_f, "release": rel_f, "win": win})
    return shots


def cut_clips(video_path: Path, shots: list, out_dir: Path, fps: float, pad_s: float = 0.5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pad = int(pad_s * fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    spans = [(max(0, s["start"] - pad), s["end"] + pad, out_dir / f"{video_path.stem}_tiro{n}.mp4")
             for n, s in enumerate(shots, 1)]
    writers, idx = {}, 0
    last = max(b for _, b, _ in spans)
    while idx <= last:
        ok, img = cap.read()
        if not ok:
            break
        for a, b, path in spans:
            if a <= idx <= b:
                if path not in writers:
                    writers[path] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writers[path].write(img)
        idx += 1
    cap.release()
    for wr in writers.values():
        wr.release()
    return list(writers)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def fps_of(df: pd.DataFrame) -> float:
    dt = np.diff(df["timestamp_s"].values)
    return 1.0 / np.median(dt[dt > 0]) if (dt > 0).any() else 30.0


# --------------------------------------------------------------------------
# Analisis de un clip completo (lo usa la app)
# --------------------------------------------------------------------------
def upload_paths(data: bytes, suffix: str, root: Path):
    """Rutas (video, cache) para un video subido; el nombre sale del contenido, asi el mismo video no se reprocesa."""
    h = hashlib.md5(data).hexdigest()[:16]
    return root / "uploads" / f"{h}{suffix.lower()}", root / "cache" / f"{h}.csv", h


def analyze_clip(video_path: Path, cache_path: Path, arm: str, rule=None, complexity: int = 1, frac: float = 0.90,
                 prep_level: float = 0.60, progress=None, make_tiles: bool = True) -> dict:
    """
    Detecta los tiros de un video (corto o largo) y mide el codo en cada uno.
    Devuelve {fps, n_frames, detection_rate, note, shots:[{n, value, verdict, has_pause, pause_s, set_s, release_s,
    flare_lift_max, flare_lift_mean, features, tile}]}. `tile` es una imagen BGR (o None).
    """
    df = E.extract_angles(video_path, cache_path, complexity, False, progress)
    det, sm = E._smoothed(df)
    fps = fps_of(df)
    info = {"fps": fps, "n_frames": len(df), "duration_s": len(df) / fps, "note": "", "shots": [],
            "detection_rate": 100 * len(det) / max(len(df), 1)}
    if len(det) < E.MIN_DETECTED_FRAMES:
        info["note"] = "Casi no se detecto a la persona en el video: revisa que se vea el cuerpo completo y bien iluminado."
        return info
    shots = find_shots(sm, arm, fps, frac, use_elbow=False)
    if not shots:   # clip ya recortado a un tiro: se analiza completo
        f0 = elbow_features(sm, arm, frac, prep_level)
        if f0 is None:
            info["note"] = ("No pude ubicar un tiro en este video. Revisa que se vea al jugador de frente, con el "
                            "cuerpo completo, y que el video incluya cuando sube el balon.")
            return info
        shots = [{"start": int(sm.index.min()), "end": int(sm.index.max()), "win": sm}]
        info["note"] = "No detecte un tiro claro dentro del video, asi que analice el clip completo como un solo tiro."
    feat = rule["feature"] if rule else HEADLINE
    measured = [(n, elbow_features(sh["win"], arm, frac, prep_level)) for n, sh in enumerate(shots, 1)]
    measured = [(n, f) for n, f in measured if f is not None]
    # una sola lectura del video para todas las imagenes (leerlo por cada tiro seria muy lento)
    pre = grab_frames(video_path, [f["peak_frame"] for _, f in measured]) if (make_tiles and measured) else None
    for n, f in measured:
        value = f[feat]
        info["shots"].append({
            "n": n, "value": float(value), "verdict": verdict(value, rule) if rule else "?",
            "has_pause": bool(f["has_pause"]), "pause_s": f["gap_frames"] / fps,
            "set_s": f["set_frame"] / fps, "release_s": f["release_frame"] / fps,
            "flare_lift_max": f["flare_lift_max"], "flare_lift_mean": f["flare_lift_mean"],
            "features": f,
            "tile": frame_tile(video_path, sm, f, "peak", f"tiro {n}", arm, preloaded=pre) if pre else None})
    if not info["shots"]:
        info["note"] = (info["note"] + " " if info["note"] else "") + "Detecte movimiento pero no pude medir el codo."
    return info


def summarize_shots(shots: list) -> dict:
    """Resumen de una sesion: cuantos tiros, apertura promedio, cuantos buenos, cuantos con pausa."""
    if not shots:
        return {}
    vals = np.array([s["value"] for s in shots], float)
    vs = [s["verdict"] for s in shots]
    return {"n": len(shots), "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
            "n_bueno": vs.count("bueno"), "n_dudoso": vs.count("dudoso"), "n_malo": vs.count("malo"),
            "pct_bueno": 100 * vs.count("bueno") / len(vs),
            "pct_pausa": 100 * float(np.mean([s["has_pause"] for s in shots]))}


def main():
    ap = argparse.ArgumentParser(description="Mide el codo abierto antes del tiro (vista de frente)")
    ap.add_argument("--arm", required=True, choices=["left", "right"], help="Brazo con el que tira la persona")
    ap.add_argument("--videos", help="Carpeta con videos etiquetados (bueno/codo en el nombre o la carpeta)")
    ap.add_argument("--test", help="Carpeta con videos de otras personas (sin etiquetas). Usalo JUNTO con --videos")
    ap.add_argument("--check", metavar="CARPETA",
                    help="Muestra, para cada video de la carpeta, el set point, el release y el momento de mayor "
                         "apertura del codo, para verificar a ojo que se identifican bien")
    ap.add_argument("--compare", nargs=2, metavar=("ANTES", "DESPUES"),
                    help="Dos videos o dos carpetas (varios tiros por lado es mas confiable)")
    ap.add_argument("--min-change", type=float, default=0.10,
                    help="Cambio minimo del pico de flare (anchos de hombro) para decir mejoro/empeoro. "
                         "Provisional: la variacion normal entre tiros de la misma persona fue ~0.1")
    ap.add_argument("--find-shots", metavar="VIDEO", help="Detecta los tiros de un video largo")
    ap.add_argument("--cut", action="store_true", help="Con --find-shots: guarda un clip por tiro")
    ap.add_argument("--shot-height", type=float, default=0.5,
                    help="Con --find-shots: altura minima de la muneca sobre el hombro (largos de torso). "
                         "Bajalo si se pierden tiros, subelo si detecta tiros falsos")
    ap.add_argument("--shot-rise", type=float, default=0.5,
                    help="Con --find-shots: cuanto debe subir la muneca en los 2 s previos (largos de torso)")
    ap.add_argument("--view", default="frente", choices=["frente", "lado"],
                    help="Desde donde se grabo (solo afecta a --find-shots)")
    ap.add_argument("--frac", type=float, default=0.90, help="Fraccion del recorrido de la muneca que marca el release")
    ap.add_argument("--prep-level", type=float, default=0.60,
                    help="Fraccion del recorrido de la muneca donde termina la 'preparacion' (ventana *_prep_*)")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--force", action="store_true", help="Reprocesar aunque exista cache")
    ap.add_argument("--output", default="elbow_out")
    args = ap.parse_args()
    if not any([args.videos, args.test, args.compare, args.find_shots, args.check]):
        ap.error("Indica al menos uno: --videos, --test, --compare, --check o --find-shots")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    report, rule, train = ["# Analisis del codo antes del tiro\n"], None, None

    # ---- 1. videos etiquetados
    if args.videos:
        rows, peaks, rels = [], {}, {}
        for i, (p, label) in enumerate(find_labeled(Path(args.videos)), 1):
            print(f"[{i}] {label:6s} {p.name}")
            res = process_video(p, out_dir, args)
            if res is None:
                print("     [!] no se pudo ubicar el tiro / poca deteccion; se excluye")
                continue
            _, sm, f = res
            rows.append({"video": p.name, "label": label, **f})
            for which, store_ in (("peak", peaks), ("release", rels)):
                t = frame_tile(p, sm, f, which, f"{p.stem[:34]} f{f[which + '_frame']}", args.arm)
                if t is not None:
                    store_.setdefault(label, []).append(t)
        if not rows:
            sys.exit("\n[X] Ningun video etiquetado se pudo analizar.")
        train = pd.DataFrame(rows)
        train.to_csv(out_dir / "elbow_features.csv", index=False)
        text, rule = analyze_labeled(train)
        report.append(text)
        fr = train[["video", "label", "start_frame", "set_frame", "release_frame", "peak_frame", "has_pause"]]
        report.append("### Frames detectados por video\n\n" + fr.to_markdown(index=False) + "\n")
        nop = int((~train["has_pause"]).sum())
        if nop:
            report.append(f"_{nop} de {len(train)} videos no tienen pausa clara en el set point (set y release casi en el "
                          "mismo frame, tiro continuo): en esos, las metricas `*_set` y `*_pre_*` no son confiables; "
                          f"usa las `*_lift_*` (por ejemplo `{HEADLINE}`) y la imagen del pico del codo._\n")
        a, b = sheet(peaks, out_dir / "hoja_pico_codo.jpg"), sheet(rels, out_dir / "hoja_release.jpg")
        if a or b:
            report.append("Imagenes (linea cian = vertical del hombro): "
                          + ", ".join(f"`{x.name}`" for x in (a, b) if x)
                          + ". `hoja_pico_codo.jpg` muestra el momento en que el codo estuvo MAS abierto durante el "
                            "levantamiento; `hoja_release.jpg` el release detectado.\n")
        else:
            report.append("_[!] No se pudieron generar imagenes: ¿se pueden abrir los videos? (el analisis salio de la "
                          "cache)._\n")

        if rule:
            save_rule(out_dir, rule, args)
    if rule is None and any([args.test, args.compare, args.find_shots, args.check]):
        rule = load_rule(out_dir)
        if rule:
            report.append(f"_Usando la regla guardada de tu ultima corrida de --videos: `{rule['feature']}` "
                          f"{rule['direction']} {rule['t']:.2f} = bueno._\n")

    # ---- 2. otras personas
    if args.test:
        test_rows, items = [], []
        for p in list_videos(Path(args.test)):
            res = process_video(p, out_dir, args)
            if res is None:
                print(f"[!] {p.name}: no se pudo analizar")
                continue
            _, sm, f = res
            test_rows.append({"video": p.name, **f})
            items.append((p, sm, f))
        report.append("## Otras personas (sin etiquetas)\n")
        if test_rows:
            t = pd.DataFrame(test_rows)
            if rule is None:
                report.append("_[!] No hay regla con que comparar: corre primero --videos con tus videos etiquetados "
                              "(o juntos: --videos ... --test ...). Solo se muestran las metricas._\n")
            else:
                t["veredicto"] = [verdict(v, rule) for v in t[rule["feature"]]]
            t.to_csv(out_dir / "sujetos_prueba.csv", index=False)
            main_feat = rule["feature"] if rule else HEADLINE
            show = [c for c in dict.fromkeys(["video", main_feat, "flare_lift_mean", "abduction_lift_mean",
                                              "has_pause", "veredicto"]) if c in t.columns]
            if rule:
                lt = "<" if rule["direction"] == "<" else ">"
                report.append(f"Regla: `{main_feat}` {lt} {rule['t']:.2f} = bueno (dudoso: ±{rule['band']:.2f}). "
                              f"Referencia de tus videos: buenos {rule['good_mean']:.2f}±{rule['good_sd']:.2f}, "
                              f"malos {rule['bad_mean']:.2f}±{rule['bad_sd']:.2f}.\n")
            report.append(t[show].round(2).to_markdown(index=False) + "\n")
            cols = {}
            preds = list(t["veredicto"]) if "veredicto" in t.columns else ["sin regla"] * len(t)
            for (p, sm, f), pred in zip(items, preds):
                tl = frame_tile(p, sm, f, "peak", f"{p.stem[:34]} f{f['peak_frame']}", args.arm)
                if tl is not None:
                    cols.setdefault(f"veredicto: {pred}", []).append(tl)
            sp = sheet(cols, out_dir / "hoja_sujetos_prueba.jpg")
            if sp:
                report.append(f"Imagen (momento de mayor apertura del codo): `{sp.name}`\n")
        else:
            report.append("_Ningun video de prueba se pudo analizar._\n")

    # ---- 3. revisar a ojo que se identifican bien los momentos
    if args.check:
        rows, cols = [], {"set point": [], "pico codo": [], "release": []}
        for p in list_videos(Path(args.check)):
            res = process_video(p, out_dir, args)
            if res is None:
                print(f"[!] {p.name}: no se pudo ubicar el tiro")
                continue
            df_, sm, f = res
            fps = fps_of(df_)
            rows.append({"video": p.name, "inicio": f["start_frame"], "set": f["set_frame"],
                         "pico_codo": f["peak_frame"], "release": f["release_frame"],
                         "set_s": f["set_frame"] / fps, "release_s": f["release_frame"] / fps,
                         "pausa_set": f["has_pause"], HEADLINE: f[HEADLINE]})
            for key, which in (("set point", "set"), ("pico codo", "peak"), ("release", "release")):
                tl = frame_tile(p, sm, f, which, f"{p.stem[:30]} f{f[which + '_frame']}", args.arm)
                if tl is not None:
                    cols[key].append(tl)
        report.append("## Revision de momentos del tiro\n")
        if rows:
            report.append(pd.DataFrame(rows).round(2).to_markdown(index=False) + "\n")
            sp = sheet(cols, out_dir / "hoja_check.jpg", cell_h=380)
            if sp:
                report.append(f"Imagen: `{sp.name}` (una fila por video; columnas: set point, pico codo (codo mas abierto), "
                              "release). Anota los videos donde un frame este mal.\n")

    # ---- 4. comparar antes / despues (videos o carpetas)
    if args.compare:
        sides = []
        for a_ in args.compare:
            pth = Path(a_)
            vids = list_videos(pth) if pth.is_dir() else [resolve_video(pth)]
            items = []
            for v in vids:
                res = process_video(v, out_dir, args)
                if res is None:
                    print(f"[!] {v.name}: no se pudo analizar")
                    continue
                items.append((v, *res))
            if not items:
                sys.exit(f"\n[X] No se pudo analizar ningun video de '{a_}'.")
            sides.append(items)

        def stat(items, m):
            v = np.array([it[3][m] for it in items], float)
            return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))

        lines = ["## Comparacion antes / despues\n",
                 f"Tiros analizados: antes {len(sides[0])}, despues {len(sides[1])}\n",
                 "| metrica | antes | despues | cambio |", "|:--|--:|--:|--:|"]
        main_feat = rule["feature"] if rule else HEADLINE
        for m in dict.fromkeys([main_feat, HEADLINE, "flare_lift_mean", "abduction_lift_mean", "elbow_vs_wrist_lift_max"]):
            (ma, sa), (mb, sb) = stat(sides[0], m), stat(sides[1], m)
            sd = lambda x: "" if np.isnan(x) else f"±{x:.2f}"
            lines.append(f"| {m} | {ma:.2f}{sd(sa)} | {mb:.2f}{sd(sb)} | {mb - ma:+.2f} |")
        d = stat(sides[1], main_feat)[0] - stat(sides[0], main_feat)[0]
        cambio_txt = ("MEJORO: el codo queda mas cerca del cuerpo" if d <= -args.min_change else
                      "EMPEORO: el codo se abre mas" if d >= args.min_change else "SIN CAMBIO claro")
        lines.append(f"\n**{cambio_txt}** (cambio del pico de flare: {d:+.2f} anchos de hombro; umbral provisional "
                     f"{args.min_change})\n")
        if len(sides[0]) == 1 and len(sides[1]) == 1:
            lines.append("_Con un solo tiro por lado el resultado es ruidoso; lo normal es que dos tiros de la misma "
                         "persona difieran ~0.1. Compara carpetas con varios tiros._\n")
        report.append("\n".join(lines))
        cols = {}
        for name, items in zip(("antes", "despues"), sides):
            for v, _, sm, f in items[:3]:
                tl = frame_tile(v, sm, f, "peak", f"{name} {v.stem[:28]} f{f['peak_frame']}", args.arm)
                if tl is not None:
                    cols.setdefault(name, []).append(tl)
        sp = sheet(cols, out_dir / "comparacion.jpg")
        if sp:
            report.append(f"Imagen (momento de mayor apertura del codo): `{sp.name}`\n")

    # ---- 5. tiros en video largo
    if args.find_shots:
        p = resolve_video(args.find_shots)
        cache = out_dir / "cache" / f"{p.parent.name}__{p.stem}.csv"
        df = E.extract_angles(p, cache, args.complexity, args.force)
        det, sm = E._smoothed(df)
        fps = fps_of(df)
        shots = find_shots(sm, args.arm, fps, args.frac, use_elbow=(args.view == "lado"),
                           min_height=args.shot_height, min_rise=args.shot_rise)
        lines = [f"## Tiros encontrados en {p.name}\n",
                 f"Video de {len(df) / fps:.0f} s ({len(df)} frames, {fps:.0f} fps), persona detectada en "
                 f"{100 * len(det) / max(len(df), 1):.0f}% de los frames. **Tiros: {len(shots)}**\n"]
        if shots:
            rows, tiles, flares = [], [], []
            for n, sh in enumerate(shots, 1):
                f = elbow_features(sh["win"], args.arm, args.frac, args.prep_level)
                r = {"tiro": n, "inicio_s": sh["start"] / fps, "set_s": sh["set"] / fps,
                     "release_s": sh["release"] / fps}
                if f:
                    r.update({HEADLINE: f[HEADLINE], "flare_lift_mean": f["flare_lift_mean"]})
                    flares.append(f[HEADLINE])
                    if rule:
                        r["regla"] = apply_rule(f[rule["feature"]], rule)
                    t_ = frame_tile(p, sm, f, "peak", f"tiro {n} f{f['peak_frame']}", args.arm)
                    if t_ is not None:
                        tiles.append(t_)
                rows.append(r)
            tb = pd.DataFrame(rows)
            tb.to_csv(out_dir / f"tiros_{p.stem}.csv", index=False)
            lines.append(tb.round(2).to_markdown(index=False) + "\n")
            if flares:
                lines.append(f"Promedio de la sesion: {HEADLINE} = {np.mean(flares):.2f} "
                             f"(sd {np.std(flares, ddof=1) if len(flares) > 1 else float('nan'):.2f}, {len(flares)} tiros)\n")
            if tiles:
                sp = sheet({f"tiro {i + 1}": [t_] for i, t_ in enumerate(tiles[:8])}, out_dir / f"tiros_{p.stem}.jpg")
                if sp:
                    lines.append(f"Imagen del momento de mayor apertura del codo de cada tiro: `{sp.name}`\n")
            if args.cut:
                clips = cut_clips(p, shots, out_dir / "clips", fps)
                lines.append("Clips guardados en `clips/` (puedes revisarlos con --check): "
                             + ", ".join(c.name for c in clips) + "\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            h = E.wrist_height(sm, args.arm)
            fig, ax = plt.subplots(figsize=(14, 3.2))
            if h is not None:
                ax.plot(h.index / fps, h.values, color="tab:blue", lw=1)
            for n, sh in enumerate(shots, 1):
                ax.axvspan(sh["start"] / fps, sh["end"] / fps, color="orange", alpha=0.15)
                ax.axvline(sh["set"] / fps, color="cyan", ls=":")
                ax.axvline(sh["release"] / fps, color="green", ls="--")
                ax.text(sh["start"] / fps, ax.get_ylim()[1], f"{n}", va="top", fontsize=9)
            ax.set_xlabel("segundos")
            ax.set_ylabel("altura de muneca\n(largos de torso)")
            ax.set_title("Tiros detectados (naranja), set point (cian), release (verde)")
            fig.tight_layout()
            fig.savefig(out_dir / f"tiros_{p.stem}.png", dpi=100)
            plt.close(fig)
            lines.append(f"Grafico: `tiros_{p.stem}.png`\n")
        except Exception as e:  # el grafico no debe tumbar el analisis
            print("[!] no se pudo hacer el grafico:", e)
        report.append("\n".join(lines))

    text = "\n".join(report)
    (out_dir / "reporte_codo.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nListo. Resultados en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
