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
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import experiment as E
from src.angle_calculator import LANDMARK_IDS

FRONT_METRICS = ["flare", "elbow_vs_wrist", "abduction"]   # tilt se calcula pero no entra a FEATS (se invierte con la muneca baja)
WINDOWS = ("set", "pre_max", "pre_mean", "lift_max", "lift_mean")
FEATS = [f"{m}_{k}" for m in FRONT_METRICS for k in WINDOWS]
HEADLINE = "flare_lift_max"   # pico de apertura del codo durante todo el levantamiento
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


def elbow_features(sm: pd.DataFrame, arm: str, frac: float = 0.90):
    """
    Metricas del codo en la fase PREVIA al tiro. Ventanas:
      set        = alrededor del set point (+-2 frames)
      pre_*      = desde que el balon empieza a subir hasta el set point
      lift_*     = desde que el balon empieza a subir hasta el release (NO depende de ubicar bien el set point)
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

    def win(a, b):
        return series[(series.index >= a) & (series.index <= b)]

    pre, lift, around = win(start_f, set_f), win(start_f, end_f), win(set_f - 2, set_f + 2)
    if len(pre) < 3 or len(lift) < 3 or len(around) == 0 or lift["flare"].dropna().empty:
        return None
    out = {"has_pause": bool(rel_f - set_f >= 3),
           "start_frame": int(start_f), "set_frame": int(set_f), "release_frame": int(rel_f),
           "peak_frame": int(lift["flare"].idxmax()), "tilt_set": float(around["tilt"].median())}
    for m in FRONT_METRICS:
        out[f"{m}_set"] = float(around[m].median())
        out[f"{m}_pre_max"], out[f"{m}_pre_mean"] = float(pre[m].max()), float(pre[m].mean())
        out[f"{m}_lift_max"], out[f"{m}_lift_mean"] = float(lift[m].max()), float(lift[m].mean())
    return out


def process_video(path: Path, out_dir: Path, args):
    cache = out_dir / "cache" / f"{path.parent.name}__{path.stem}.csv"
    df = E.extract_angles(path, cache, args.complexity, args.force)
    det, sm = E._smoothed(df)
    if len(det) < E.MIN_DETECTED_FRAMES:
        return None
    f = elbow_features(sm, args.arm, args.frac)
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


def frame_tile(path, sm, f, which, caption, arm):
    """Imagen recortada de un momento del tiro: which = 'peak' (codo mas abierto), 'set' o 'release'."""
    frame = f[f"{which}_frame"]
    imgs, (w, h) = grab_frames(path, [frame])
    if not imgs:
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
                 "tilt y abduction en grados.\n")
    lines.append(table.head(8).round(2).to_markdown(index=False) + "\n")

    # La regla solo puede usar metricas del levantamiento completo (*_lift_*): no dependen de acertar el set point,
    # asi que tambien valen para quien tira en un solo movimiento, sin pausa.
    lift = table[table["metrica"].str.contains("_lift_")]
    head = lift[lift["metrica"] == HEADLINE]
    # se prefiere la metrica principal (codo hacia afuera) si separa bien: es la mas facil de explicar
    best = HEADLINE if (len(head) and head.iloc[0]["AUC"] >= 0.9) else lift.iloc[0]["metrica"]
    t, direction, acc = E.best_threshold(good[best].values, bad[best].values)
    rule = {"feature": best, "t": t, "direction": direction}
    lines.append("### Regla candidata (una sola metrica)\n")
    lines.append(f"- `{best}` {direction} {t:.2f} = tiro bueno (acierta {acc * 100:.0f}% de {len(df)} videos)\n")
    from math import comb
    p_perfect = 2 / comb(len(good) + len(bad), len(good))
    lines.append(f"_Ojo: con {len(good)} vs {len(bad)} videos, una metrica que no sirve de nada saca AUC 1.0 el "
                 f"{p_perfect:.1%} de las veces, y aqui se miraron {len(FEATS)}. Por eso lo que cuenta es que "
                 f"varias formas distintas de medir lo mismo coincidan, y que se repita con otras personas._\n")

    X = df[FEATS].values.astype(float)
    preds = E.loo_predictions(X, y)
    acc_loo = float((preds == y).mean())
    base = float((y == "malo").mean())
    cm = pd.crosstab(pd.Series(y, name="real"), pd.Series(preds, name="predicho"))
    lines.append("### Clasificador (leave-one-out)\n")
    lines.append(f"- Acierto bueno vs malo: **{acc_loo * 100:.0f}%**  (decir siempre 'malo': {base * 100:.0f}%)\n")
    lines.append(cm.to_markdown() + "\n")
    wrong = df[preds != y]
    if len(wrong):
        lines.append("Mal clasificados: " + ", ".join(f"{v} ({l}→{p})" for v, l, p in
                                                     zip(wrong["video"], wrong["label"], preds[preds != y])) + "\n")
    return "\n".join(lines), rule


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
            have_train = train is not None and (train["label"] == "bueno").any() and (train["label"] != "bueno").any()
            if have_train:
                ytr = np.where(train["label"] == "bueno", "bueno", "malo")
                t["clasificador"] = E.fit_predict(train[FEATS].values.astype(float), ytr, t[FEATS].values.astype(float))
                if rule:
                    t["regla"] = [apply_rule(v, rule) for v in t[rule["feature"]]]
            else:
                report.append("_[!] Corriste --test sin --videos: no hay videos etiquetados con que comparar, asi que "
                              "solo se muestran las metricas. Usa los dos juntos._\n")
            t.to_csv(out_dir / "sujetos_prueba.csv", index=False)
            show = [c for c in ["video", HEADLINE, "flare_lift_mean", "abduction_lift_mean", "flare_set",
                                "clasificador", "regla"] if c in t.columns]
            if have_train:
                g, bd = train[train["label"] == "bueno"], train[train["label"] != "bueno"]
                report.append("Referencia (tus videos): " + "; ".join(
                    f"{c}: buenos {g[c].mean():.2f}±{g[c].std():.2f}, malos {bd[c].mean():.2f}±{bd[c].std():.2f}"
                    for c in show[1:4]) + "\n")
            report.append(t[show].round(2).to_markdown(index=False) + "\n")
            cols = {}
            preds = list(t["clasificador"]) if "clasificador" in t.columns else ["sin comparar"] * len(t)
            for (p, sm, f), pred in zip(items, preds):
                tl = frame_tile(p, sm, f, "peak", f"{p.stem[:34]} f{f['peak_frame']}", args.arm)
                if tl is not None:
                    cols.setdefault(f"predicho: {pred}", []).append(tl)
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
        for m in (HEADLINE, "flare_lift_mean", "abduction_lift_mean", "elbow_vs_wrist_lift_max", "flare_set"):
            (ma, sa), (mb, sb) = stat(sides[0], m), stat(sides[1], m)
            sd = lambda x: "" if np.isnan(x) else f"±{x:.2f}"
            lines.append(f"| {m} | {ma:.2f}{sd(sa)} | {mb:.2f}{sd(sb)} | {mb - ma:+.2f} |")
        d = stat(sides[1], HEADLINE)[0] - stat(sides[0], HEADLINE)[0]
        verdict = ("MEJORO: el codo queda mas cerca del cuerpo" if d <= -args.min_change else
                   "EMPEORO: el codo se abre mas" if d >= args.min_change else "SIN CAMBIO claro")
        lines.append(f"\n**{verdict}** (cambio del pico de flare: {d:+.2f} anchos de hombro; umbral provisional "
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
                f = elbow_features(sh["win"], args.arm, args.frac)
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
