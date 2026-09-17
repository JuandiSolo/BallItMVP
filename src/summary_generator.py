"""
summary_generator.py
-----------------------
Convierte los datos crudos (angulo por frame, que pueden ser cientos de
numeros) en un resumen corto y legible: minimos, maximos, y en que
momento (frame/segundo) ocurrieron. Este resumen es lo que le pasamos
a la IA -- mandarle todos los frames crudos gastaria tokens y no
aportaria mas que el resumen bien hecho.
"""

def build_summary(frames_data: list, fps: float) -> dict:
    """
    Devuelve un diccionario con, para cada angulo:
      - valor minimo y en que segundo ocurrio
      - valor maximo y en que segundo ocurrio
      - rango de movimiento (max - min)
    Solo se consideran los frames donde SI se detecto a la persona.
    """
    detected_frames = [f for f in frames_data if f["detected"]]
    if not detected_frames:
        return {}

    angle_names = detected_frames[0]["angles"].keys()
    summary = {}

    for angle_name in angle_names:
        # Filtramos frames donde ese angulo especifico se pudo calcular
        valid = [
            (f["timestamp_s"], f["angles"][angle_name])
            for f in detected_frames
            if f["angles"].get(angle_name) is not None
        ]
        if not valid:
            continue

        min_t, min_v = min(valid, key=lambda x: x[1])
        max_t, max_v = max(valid, key=lambda x: x[1])

        summary[angle_name] = {
            "min_value": min_v,
            "min_at_second": round(min_t, 2),
            "max_value": max_v,
            "max_at_second": round(max_t, 2),
            "range_of_motion": round(max_v - min_v, 1),
        }

    return summary


def summary_to_markdown(summary: dict, movement_label: str = "movimiento de basquet") -> str:
    """
    Convierte el resumen numerico a texto en Markdown, legible para humanos
    y listo para pegar como parte de un prompt a una API de IA (Claude, etc).
    """
    lines = [f"# Resumen del análisis de {movement_label}", ""]
    lines.append("| Articulación | Mínimo | Máximo | Rango de movimiento |")
    lines.append("|---|---|---|---|")

    readable_names = {
        "left_elbow_angle": "Codo izquierdo",
        "right_elbow_angle": "Codo derecho",
        "left_knee_angle": "Rodilla izquierda",
        "right_knee_angle": "Rodilla derecha",
        "left_hip_angle": "Cadera izquierda",
        "right_hip_angle": "Cadera derecha",
        "left_shoulder_angle": "Hombro izquierdo",
        "right_shoulder_angle": "Hombro derecho",
    }

    for angle_name, stats in summary.items():
        label = readable_names.get(angle_name, angle_name)
        lines.append(
            f"| {label} | {stats['min_value']}° (t={stats['min_at_second']}s) "
            f"| {stats['max_value']}° (t={stats['max_at_second']}s) "
            f"| {stats['range_of_motion']}° |"
        )

    return "\n".join(lines)


def build_ai_prompt(summary_markdown: str, movement_label: str = "tiro de basquet") -> str:
    """
    Arma el prompt final que le mandarias a la API de Claude (o cualquier
    LLM) para pedir el analisis tecnico. Aqui es donde defines el "rol"
    del analisis: que sea coach de basquet, biomecanica, etc.
    """
    return f"""Eres un entrenador de básquet con conocimientos de biomecánica.
Te voy a dar datos de ángulos articulares extraídos de un video de un(a)
jugador(a) haciendo un {movement_label}. Los datos vienen de un sistema de
seguimiento de pose (MediaPipe), calculados en grados, con el mínimo, máximo
y rango de movimiento de cada articulación durante el movimiento.

Con base en estos datos:
1. Identifica si los ángulos están dentro de rangos técnicamente saludables/eficientes.
2. Señala qué articulación muestra el patrón más atípico o riesgoso.
3. Da 2-3 recomendaciones concretas y accionables para mejorar la técnica.

Datos del movimiento:

{summary_markdown}
"""
