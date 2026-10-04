"""
app.py - BallIt: revision del codo en el tiro libre (prototipo de escritorio)

Correr (desde la raiz del repo):
    pip install -r requirements.txt
    streamlit run app.py

Usa la regla que calculaste con:  python3 analyze_elbow.py --videos frente/ --arm right
(se guarda en elbow_out/rule.json). Si todavia no existe, usa una regla de ejemplo.
"""

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import analyze_elbow as A

ROOT = Path(__file__).parent
APP_CACHE = ROOT / "app_cache"
OUT_DIR = ROOT / "elbow_out"
VIDEO_TYPES = ["mp4", "mov", "m4v", "avi", "mkv"]

st.set_page_config(page_title="BallIt - codo en tiro libre", page_icon="🏀", layout="wide")

if "results" not in st.session_state:
    st.session_state.results = {}
if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------------------------------
# Regla
# --------------------------------------------------------------------------
def load_rule():
    saved = A.load_rule(OUT_DIR)
    return (saved, True) if saved else (dict(A.DEFAULT_RULE), False)


rule, rule_saved = load_rule()
FEATURE_NAMES = {
    "flare_lift_max": "apertura máxima del codo (todo el levantamiento)",
    "flare_lift_mean": "apertura promedio del codo (todo el levantamiento)",
    "flare_prep_max": "apertura máxima del codo en la preparación",
    "flare_prep_mean": "apertura promedio del codo en la preparación",
}
FEATURE_NAMES.update({
    "abduction_prep_mean": "ángulo promedio del brazo respecto al torso en la preparación",
    "abduction_prep_max": "ángulo máximo del brazo respecto al torso en la preparación",
    "abduction_lift_mean": "ángulo promedio del brazo respecto al torso durante el levantamiento",
    "abduction_lift_max": "ángulo máximo del brazo respecto al torso durante el levantamiento",
})
feature_label = FEATURE_NAMES.get(rule["feature"], rule["feature"])
is_angle = rule["feature"].startswith("abduction_")
unit = "grados" if is_angle else "anchos de hombro"
symbol = "°" if is_angle else " anchos de hombro"
good_is_lower = rule["direction"] == "<"


BADGE = {"bueno": ("🟢", "Codo bien alineado"), "dudoso": ("🟡", "Cerca del límite"),
         "malo": ("🔴", "Codo muy abierto"), "?": ("⚪", "Sin regla")}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")
    arm_label = st.radio("¿Con qué mano tira la persona?", ["Derecha", "Izquierda"], horizontal=True)
    arm = "right" if arm_label == "Derecha" else "left"
    st.caption("Si en las imágenes el esqueleto verde sale en el otro brazo, cambia esta opción.")
    if is_angle:
        initial_change = round(min(20.0, max(1.0, float(rule.get("band", 4.0)))), 1)
        min_change = st.slider(f"Cambio mínimo para comparar ({unit})", 1.0, 20.0, initial_change, 0.1,
                               help="Sensibilidad provisional: usa inicialmente la banda de la regla. No es una medida validada de variación entre tiros.")
    else:
        min_change = st.slider(f"Cambio mínimo para comparar ({unit})", 0.03, 0.30, 0.10, 0.01,
                               help="Sensibilidad provisional. Compara varios tiros con la misma cámara y encuadre.")
    st.divider()
    st.subheader("Cómo se decide")
    meaning = ("Ángulo 2D formado por cadera, hombro y codo." if is_angle else
               "0 = codo bajo el hombro; 1 = un ancho de hombros hacia afuera.")
    low, high = rule['t'] - rule['band'], rule['t'] + rule['band']
    good_limit, bad_limit = (low, high) if good_is_lower else (high, low)
    good_word, bad_word = ("menos", "más") if good_is_lower else ("más", "menos")
    st.markdown(
        f"Se mide **{feature_label}**, en **{unit}**. {meaning}\n\n"
        f"- 🟢 Bien: {good_word} de **{good_limit:.2f}{symbol}**\n"
        f"- 🟡 Dudoso: alrededor del límite, entre {low:.2f}{symbol} y {high:.2f}{symbol}\n"
        f"- 🔴 Mal: {bad_word} de **{bad_limit:.2f}{symbol}**")
    st.caption(f"Referencia: tiros buenos {rule['good_mean']:.2f}±{rule['good_sd']:.2f}, "
               f"tiros con codo abierto {rule['bad_mean']:.2f}±{rule['bad_sd']:.2f} "
               f"({rule['n_good']} y {rule['n_bad']} videos).")
    if not rule_saved:
        st.warning("Todavía no calculaste tu regla: se usa una de ejemplo. Corre "
                   "`python3 analyze_elbow.py --videos frente/ --arm right` y recarga la página.")
    with st.expander("📹 Cómo grabar"):
        st.markdown("- **De frente**, mirando a la cámara (no de lado ni en diagonal).\n"
                    "- Cuerpo completo dentro del cuadro, cámara fija a la altura del pecho.\n"
                    "- Buena luz y una sola persona.\n"
                    "- Que el video incluya cuando **sube el balón**. Puede tener varios tiros.")
    st.info("Prototipo: mide solo el **codo** y solo desde el **frente**. Calibrado con pocos videos de una persona; "
            "no es un diagnóstico.")


# --------------------------------------------------------------------------
# Helpers de analisis y presentacion
# --------------------------------------------------------------------------
def run_upload(up):
    data = up.getvalue()
    video, cache, h = A.upload_paths(data, Path(up.name).suffix or ".mp4", APP_CACHE)
    key = (h, arm, tuple(sorted((k, repr(v)) for k, v in rule.items())))
    if key in st.session_state.results:
        return st.session_state.results[key], key
    video.parent.mkdir(parents=True, exist_ok=True)
    if not video.exists():
        video.write_bytes(data)
    bar = st.progress(0.0, text=f"Analizando {up.name} …")
    try:
        info = A.analyze_clip(video, cache, arm, rule, frac=float(rule.get("frac", 0.90)),
                              prep_level=float(rule.get("prep_level", 0.60)), progress=lambda x: bar.progress(x, text=f"Analizando {up.name} …"))
    except Exception as e:  # un video corrupto no debe tumbar la app
        bar.empty()
        st.error(f"No pude analizar «{up.name}»: {e}")
        return None, key
    bar.empty()
    info["name"] = up.name
    info["summary"] = A.summarize_shots(info["shots"])
    st.session_state.results[key] = info
    return info, key


def to_rgb(tile):
    return None if tile is None else tile[:, :, ::-1]


def advice(shot):
    out = []
    if shot["verdict"] == "malo":
        out.append(f"**Revisa el codo:** {feature_label}: {shot['value']:.2f}{symbol}. La medición queda del lado "
                   "de los ejemplos etiquetados con codo abierto. Revisa su alineación debajo del balón.")
    elif shot["verdict"] == "dudoso":
        out.append("**Cerca del límite:** el codo está algo abierto. Vigila que quede debajo del balón.")
    elif shot["verdict"] == "bueno":
        out.append("**Buen codo:** queda alineado con el hombro antes del tiro.")
    if not shot["has_pause"]:
        out.append("**Haz una pausa:** tiraste de corrido. Detente un instante con el balón arriba (set point) antes de "
                   "extender el brazo; ayuda a acomodar el codo.")
    return out


def show_quality(info):
    if info["note"]:
        st.warning(info["note"])
    if info["detection_rate"] < 80:
        st.warning(f"Solo se detectó a la persona en el {info['detection_rate']:.0f}% del video: el resultado puede ser poco fiable.")


def show_shots(info):
    for s in info["shots"]:
        icon, name = BADGE.get(s["verdict"], BADGE["?"])
        c1, c2 = st.columns([1, 2])
        with c1:
            if s["tile"] is not None:
                st.image(to_rgb(s["tile"]), width=280)
            else:
                st.caption("(no se pudo dibujar la imagen)")
        with c2:
            st.markdown(f"### {icon} Tiro {s['n']} · {name}")
            st.markdown(f"{feature_label.capitalize()}: **{s['value']:.2f}{symbol}** "
                        f"· pausa en el set point: **{'sí' if s['has_pause'] else 'no'}**"
                        f" · momento: {s['release_s']:.1f} s")
            for m in advice(s):
                st.markdown(f"- {m}")
        st.divider()


def bars(sides: dict, title):
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = {"bueno": "#2e9e4f", "dudoso": "#e0b400", "malo": "#d64545", "?": "#888888"}
    x0, ticks, labels = 0, [], []
    for name, info in sides.items():
        for s in info["shots"]:
            ax.bar(x0, s["value"], color=colors.get(s["verdict"], "#888"), width=0.8)
            ticks.append(x0)
            labels.append(f"{name[:1].upper()}{s['n']}")
            x0 += 1
        x0 += 1
    ax.axhline(rule["t"], color="k", ls="--", lw=1)
    ax.text(ax.get_xlim()[1], rule["t"], " límite", va="bottom", ha="right", fontsize=8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(f"Medición del codo\n({unit})")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def summary_row(sm):
    c = st.columns(4)
    c[0].metric("Tiros detectados", sm["n"])
    c[1].metric("Codo bien", f"{sm['n_bueno']} de {sm['n']}")
    c[2].metric("Apertura promedio", f"{sm['mean']:.2f}{symbol}")
    c[3].metric("Con pausa", f"{sm['pct_pausa']:.0f}%")


def save_history(info, key, label):
    if any(h["clave"] == key for h in st.session_state.history):
        st.info("Este análisis ya está en el historial.")
        return
    sm = info["summary"]
    st.session_state.history.append({
        "clave": key, "hora": datetime.now().strftime("%d/%m %H:%M"), "video": label, "tiros": sm["n"],
        "tiros_bien": sm["n_bueno"], "pct_bien": round(sm["pct_bueno"]), "apertura_promedio": round(sm["mean"], 3),
        "pct_pausa": round(sm["pct_pausa"]), "metrica": rule["feature"], "unidad": unit})
    st.success("Guardado en el historial.")


# --------------------------------------------------------------------------
# Pagina
# --------------------------------------------------------------------------
st.title("🏀 BallIt · revisa tu codo en el tiro libre")
st.caption("Sube un video grabado de frente. La app encuentra tus tiros, mide qué tan abierto va el codo ANTES de "
           "tirar y te dice qué mejorar.")
tab1, tab2, tab3 = st.tabs(["1 · Analizar un video", "2 · Comparar antes y después", "3 · Historial"])

# ---- Tab 1
with tab1:
    up = st.file_uploader("Video de frente (puede tener varios tiros)", type=VIDEO_TYPES, key="single")
    if up:
        info, key = run_upload(up)
        if info:
            show_quality(info)
            if info["shots"]:
                summary_row(info["summary"])
                if len(info["shots"]) > 1:
                    st.pyplot(bars({"tiros": info}, "Apertura del codo por tiro"))
                show_shots(info)
                if st.button("💾 Guardar este análisis en el historial"):
                    save_history(info, key, up.name)
    else:
        st.info("Sube un video para empezar.")

# ---- Tab 2
with tab2:
    st.markdown("Sube un video de **antes** y otro de **después** (misma cámara y mismo lugar). Mientras más tiros "
                "tenga cada uno, más confiable es la comparación.")
    ca, cb = st.columns(2)
    up_a = ca.file_uploader("ANTES", type=VIDEO_TYPES, key="before")
    up_b = cb.file_uploader("DESPUÉS", type=VIDEO_TYPES, key="after")
    if up_a and up_b:
        ia, ka = run_upload(up_a)
        ib, kb = run_upload(up_b)
        if ia and ib:
            show_quality(ia)
            show_quality(ib)
            if not ia["shots"] or not ib["shots"]:
                st.error("No se pudieron medir tiros en uno de los videos.")
            else:
                sa, sb = ia["summary"], ib["summary"]
                d = sb["mean"] - sa["mean"]
                if (d if good_is_lower else -d) <= -min_change:
                    st.success(f"✅ **Cambio hacia el lado bueno de la regla.** La medición pasó de {sa['mean']:.2f}{symbol} a {sb['mean']:.2f}{symbol} "
                               f"({d:+.2f}{symbol}).")
                elif (d if good_is_lower else -d) >= min_change:
                    st.error(f"⚠️ **Cambio hacia el lado malo de la regla.** La medición pasó de {sa['mean']:.2f}{symbol} a {sb['mean']:.2f}{symbol} "
                             f"({d:+.2f}{symbol}). Revisa la alineación del codo.")
                else:
                    st.info(f"➖ **Sin cambio claro** ({d:+.2f}{symbol}; sensibilidad elegida: {min_change:.2f}{symbol}).")
                if sa["n"] == 1 or sb["n"] == 1:
                    st.caption("Con un solo tiro por lado se compara una medición por video; no demuestra una mejora estable. Prueba varios tiros por lado.")
                df = pd.DataFrame({
                    "": ["Tiros", "Codo bien", "Apertura promedio", "Con pausa"],
                    "Antes": [str(sa["n"]), f"{sa['n_bueno']} de {sa['n']}", f"{sa['mean']:.2f}", f"{sa['pct_pausa']:.0f}%"],
                    "Después": [str(sb["n"]), f"{sb['n_bueno']} de {sb['n']}", f"{sb['mean']:.2f}", f"{sb['pct_pausa']:.0f}%"]})
                st.dataframe(df, hide_index=True)
                st.pyplot(bars({"antes": ia, "despues": ib}, "Apertura del codo por tiro (A = antes, D = después)"))
                i1, i2 = st.columns(2)
                for col, name, inf in ((i1, "ANTES (tiro con más apertura)", ia), (i2, "DESPUÉS (tiro con más apertura)", ib)):
                    worst = max(inf["shots"], key=lambda s: s["value"])
                    with col:
                        st.markdown(f"**{name}** · {worst['value']:.2f}")
                        if worst["tile"] is not None:
                            st.image(to_rgb(worst["tile"]), width=300)
                if st.button("💾 Guardar ANTES y DESPUÉS en el historial"):
                    save_history(ia, ka, f"ANTES · {up_a.name}")
                    save_history(ib, kb, f"DESPUÉS · {up_b.name}")
    else:
        st.info("Sube los dos videos para comparar.")

# ---- Tab 3
with tab3:
    hist = st.session_state.history
    if not hist:
        st.info("Todavía no hay análisis guardados. Usa «Guardar en el historial» después de analizar un video.")
    else:
        df = pd.DataFrame(hist).drop(columns="clave")
        c = st.columns(4)
        c[0].metric("Análisis guardados", len(df))
        c[1].metric("Tiros totales", int(df["tiros"].sum()))
        c[2].metric("Codo bien (total)", f"{int(df['tiros_bien'].sum())} de {int(df['tiros'].sum())}")
        first, last = df["apertura_promedio"].iloc[0], df["apertura_promedio"].iloc[-1]
        c[3].metric("Cambio primera → última", f"{last - first:+.2f}", delta=f"{last - first:+.2f}", delta_color="inverse")
        st.dataframe(df, hide_index=True)
        st.line_chart(df.set_index("hora")["apertura_promedio"], height=240)
        st.caption("La tendencia usa la métrica de la regla actual; compara sesiones con la misma cámara y encuadre.")
        cc1, cc2 = st.columns(2)
        cc1.download_button("⬇️ Descargar historial (CSV)", df.to_csv(index=False).encode("utf-8"),
                            "historial_ballit.csv", "text/csv")
        if cc2.button("🗑️ Borrar historial"):
            st.session_state.history = []
            st.rerun()
