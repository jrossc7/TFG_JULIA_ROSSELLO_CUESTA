"""Interfaz y visualización del sistema.

Gestiona las ventanas de OpenCV, los controles y las gráficas finales.
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import config

# ─────────────────────────────────────────────────────────────────────────────
# VENTANAS Y CONTROLES (OPENCV)
# ─────────────────────────────────────────────────────────────────────────────

def create_debug_window() -> None:
    """Crea la ventana de depuración y sus controles."""
    cv2.namedWindow(config.WIN_DEBUG, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WIN_DEBUG, 1000, 700)

    cv2.createTrackbar(config.TRACK_THRESH, config.WIN_DEBUG, config.DEFAULT_THRESH, 255, lambda _: None)
    # El área mínima se desplaza 100 unidades respecto al control.
    cv2.createTrackbar(config.TRACK_AREA, config.WIN_DEBUG, config.DEFAULT_MIN_AREA - 100, 1900, lambda _: None)
    # La circularidad se expresa como un valor entre 0,10 y 1,00.
    cv2.createTrackbar(config.TRACK_CIRC, config.WIN_DEBUG, int(config.DEFAULT_MIN_CIRC * 100), 100, lambda _: None)

def read_trackbars() -> tuple:
    """Lee los valores actuales de los controles."""
    thresh_val   = cv2.getTrackbarPos(config.TRACK_THRESH, config.WIN_DEBUG)
    min_area     = cv2.getTrackbarPos(config.TRACK_AREA,   config.WIN_DEBUG) + 100
    min_circ_pct = cv2.getTrackbarPos(config.TRACK_CIRC,   config.WIN_DEBUG)
    min_circ_pct = max(10, min_circ_pct)
    min_circ     = min_circ_pct / 100.0
    return thresh_val, min_area, min_circ, min_circ_pct

# ─────────────────────────────────────────────────────────────────────────────
# DIBUJO EN VÍDEO (OPENCV)
# ─────────────────────────────────────────────────────────────────────────────

def fit_to_canvas(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Ajusta una imagen a un lienzo blanco conservando su proporción."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def build_main_view(vis_frame: np.ndarray,
                     canvas_w: int = config.MAIN_WINDOW_W,
                     canvas_h: int = config.MAIN_WINDOW_H) -> np.ndarray:
    """Ajusta el fotograma principal al tamaño de la ventana."""
    return fit_to_canvas(vis_frame, canvas_w, canvas_h)


def build_debug_panel(vis_frame: np.ndarray, gray_smooth: np.ndarray, binary_mask: np.ndarray, filled_mask=None,
                       panel_w: int = config.DEBUG_PANEL_W, panel_h: int = config.DEBUG_PANEL_H) -> np.ndarray:
    """Construye el panel de depuración con tres imágenes alineadas."""

    def _label(img, text):
        out = img.copy()
        cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        return out

    p1 = _label(fit_to_canvas(vis_frame, panel_w, panel_h), "Original + Tracking")
    p2 = _label(fit_to_canvas(gray_smooth, panel_w, panel_h), "Gaussiano (ROI)")

    if filled_mask is not None:
        p3 = _label(fit_to_canvas(filled_mask, panel_w, panel_h), "Mascara Rellena")
    else:
        p3 = _label(fit_to_canvas(binary_mask, panel_w, panel_h), "Mascara (sin det.)")

    return np.hstack([p1, p2, p3])

def draw_detection(frame_bgr: np.ndarray, pupil_result: dict, frame_idx: int, thresh_val: int, min_area: int, min_circ_pct: int, roi_rect: tuple = None) -> np.ndarray:
    """Dibuja la detección sobre el fotograma original."""
    vis = frame_bgr.copy()

    if roi_rect is not None:
        rx, ry, rw, rh = roi_rect
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 1)

    if pupil_result is not None:
        cx_f, cy_f = pupil_result["center"]
        # OpenCV requiere coordenadas enteras para dibujar.
        # Las coordenadas subpíxel se conservan para el análisis.
        cx, cy   = int(round(cx_f)), int(round(cy_f))
        ellipse  = pupil_result["ellipse"]

        if ellipse is not None:
            cv2.ellipse(vis, ellipse, (0, 255, 80), 2)
        else:
            radius = int(math.sqrt(pupil_result['area'] / math.pi))
            cv2.circle(vis, (cx, cy), radius, (0, 200, 255), 2)

        arm = 12
        cv2.line(vis, (cx - arm, cy), (cx + arm, cy), (0, 50, 255), 2)
        cv2.line(vis, (cx, cy - arm), (cx, cy + arm), (0, 50, 255), 2)
        cv2.circle(vis, (cx, cy), 3, (0, 50, 255), -1)

        estado = "elipse" if ellipse is not None else "centroide"
        label  = f"({cx},{cy}) [{estado}]  A={pupil_result['area']:.0f}  C={pupil_result['circularity']:.2f}"
        cv2.putText(vis, label, (cx + 14, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 80), 1, cv2.LINE_AA)
    else:
        cv2.putText(vis, "Pupila NO detectada", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2, cv2.LINE_AA)

    h, w = vis.shape[:2]
    dyn_area = pupil_result.get("dynamic_max_area", config.MAX_PUPIL_AREA_FALLBACK) if pupil_result else config.MAX_PUPIL_AREA_FALLBACK
    info = f"F#{frame_idx}  Thr={thresh_val}  Amin={min_area}  Cmin={min_circ_pct/100:.2f}  Amax*={dyn_area:.0f}"
    cv2.putText(vis, info, (w - 360, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    return vis

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICAS CLÍNICAS (MATPLOTLIB)
# ─────────────────────────────────────────────────────────────────────────────

def mostrar_graficas_clinicas(datos: dict) -> None:
    """Genera los nistagmogramas y la tabla de resultados."""

    # ── Recuperar señales ────────────────────────────────────────────────────
    t_arr = np.asarray(datos["t_arr"], dtype=float)
    x_raw_deg = np.asarray(datos["x_raw_deg"], dtype=float)
    y_raw_deg = np.asarray(datos["y_raw_deg"], dtype=float)

    # Señal suavizada usada para situar los marcadores.
    x_sg_deg = np.asarray(
        datos.get("x_sg_deg", x_raw_deg),
        dtype=float
    )

    idx_left = np.asarray(
        datos.get("idx_left", []),
        dtype=int
    )

    idx_right = np.asarray(
        datos.get("idx_right", []),
        dtype=int
    )

    idx_up = np.asarray(
        datos.get("idx_up", []),
        dtype=int
    )

    idx_down = np.asarray(
        datos.get("idx_down", []),
        dtype=int
    )

    # Señal vertical suavizada usada para situar los marcadores.
    y_sg_deg = np.asarray(
        datos.get("y_sg_deg", y_raw_deg),
        dtype=float
    )

    # Eliminar índices no válidos.
    idx_left = idx_left[
        (idx_left >= 0) &
        (idx_left < len(t_arr)) &
        (idx_left < len(x_sg_deg))
    ]

    idx_right = idx_right[
        (idx_right >= 0) &
        (idx_right < len(t_arr)) &
        (idx_right < len(x_sg_deg))
    ]

    idx_up = idx_up[
        (idx_up >= 0) &
        (idx_up < len(t_arr)) &
        (idx_up < len(y_sg_deg))
    ]

    idx_down = idx_down[
        (idx_down >= 0) &
        (idx_down < len(t_arr)) &
        (idx_down < len(y_sg_deg))
    ]

    # ── Límites gráficos ─────────────────────────────────────────────────────
    # Admite un límite común o límites independientes por eje.
    limit_default = float(
        getattr(config, "Y_AXIS_LIMIT", 25.0)
    )

    limit_x = float(
        getattr(config, "Y_AXIS_LIMIT_X", limit_default)
    )

    limit_y = float(
        getattr(config, "Y_AXIS_LIMIT_Y", 25.0)
    )

    # ── Crear figura ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"hspace": 0.35}
    )

    fig.suptitle(
        "Nistagmograma — Reporte Clínico de Posición",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    # ── Gráfica horizontal ───────────────────────────────────────────────────
    ax1.plot(
        t_arr,
        x_raw_deg,
        color="steelblue",
        linewidth=1.6,
        alpha=0.95,
        label="Posición X (cruda)",
        zorder=2
    )

    # Nistagmo izquierdo: fase rápida negativa.
    if idx_left.size > 0:
        ax1.scatter(
            t_arr[idx_left],
            x_sg_deg[idx_left],
            marker="v",
            s=45,
            color="red",
            edgecolors="black",
            linewidths=0.4,
            label=f"Nistagmo izquierdo ({len(idx_left)})",
            zorder=4
        )

    # Nistagmo derecho: fase rápida positiva.
    if idx_right.size > 0:
        ax1.scatter(
            t_arr[idx_right],
            x_sg_deg[idx_right],
            marker="^",
            s=45,
            color="darkorange",
            edgecolors="black",
            linewidths=0.4,
            label=f"Nistagmo derecho ({len(idx_right)})",
            zorder=4
        )

    ax1.set_title(
        "Nistagmo Horizontal (Eje X)",
        fontsize=11
    )
    ax1.set_ylabel(
        "Posición [°]",
        fontsize=10
    )
    ax1.grid(
        True,
        linestyle="--",
        alpha=0.40,
        zorder=1
    )
    ax1.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.50,
        zorder=1
    )

    finite_x = x_raw_deg[np.isfinite(x_raw_deg)]

    max_abs_x = (
        float(np.max(np.abs(finite_x)))
        if finite_x.size > 0
        else limit_x
    )

    ylim_h = max(
        limit_x,
        max_abs_x * 1.15
    )

    ax1.set_ylim(
        -ylim_h,
        ylim_h
    )

    ax1.legend(
        loc="upper right",
        fontsize=8.5,
        framealpha=0.92
    )

    # ── Gráfica vertical ─────────────────────────────────────────────────────
    ax2.plot(
        t_arr,
        y_raw_deg,
        color="tomato",
        linewidth=1.6,
        alpha=0.95,
        label="Posición Y (cruda)",
        zorder=2
    )

    # Nistagmo hacia abajo: fase rápida negativa.
    if idx_down.size > 0:
        ax2.scatter(
            t_arr[idx_down],
            y_sg_deg[idx_down],
            marker="v",
            s=45,
            color="red",
            edgecolors="black",
            linewidths=0.4,
            label=f"Nistagmo abajo ({len(idx_down)})",
            zorder=4
        )

    # Nistagmo hacia arriba: fase rápida positiva.
    if idx_up.size > 0:
        ax2.scatter(
            t_arr[idx_up],
            y_sg_deg[idx_up],
            marker="^",
            s=45,
            color="darkorange",
            edgecolors="black",
            linewidths=0.4,
            label=f"Nistagmo arriba ({len(idx_up)})",
            zorder=4
        )

    ax2.set_title(
        "Nistagmo Vertical (Eje Y)",
        fontsize=11
    )
    ax2.set_ylabel(
        "Posición [°]",
        fontsize=10
    )
    ax2.set_xlabel(
        "Tiempo (s)",
        fontsize=10
    )
    ax2.grid(
        True,
        linestyle="--",
        alpha=0.40,
        zorder=1
    )
    ax2.axhline(
        0,
        color="black",
        linewidth=0.8,
        alpha=0.50,
        zorder=1
    )
    ax2.legend(
        loc="upper right",
        fontsize=8.5,
        framealpha=0.92
    )

    finite_y = y_raw_deg[np.isfinite(y_raw_deg)]

    max_abs_y = (
        float(np.max(np.abs(finite_y)))
        if finite_y.size > 0
        else limit_y
    )

    ylim_v = max(
        limit_y,
        max_abs_y * 1.15
    )

    ax2.set_ylim(
        -ylim_v,
        ylim_v
    )

    # ── Tabla clínica ────────────────────────────────────────────────────────
    tabla_vfl = (
        "  TABLA DE RESULTADOS — VFL HORIZONTAL\n"
        f"  {'Dirección':<12} | "
        f"{'a.VFL (°/s)':^11} | "
        f"{'Nistagmos/s':^11} | "
        f"{'Nistagmos':^9}\n"
        f"  {'─' * 51}\n"
        f"  {'Izquierda':<12} | "
        f"{datos['avfl_left']:^11.1f} | "
        f"{datos['nps_left']:^11.2f} | "
        f"{datos['n_left']:^9}\n"
        f"  {'Derecha':<12} | "
        f"{datos['avfl_right']:^11.1f} | "
        f"{datos['nps_right']:^11.2f} | "
        f"{datos['n_right']:^9}\n"
        "\n"
        "  TABLA DE RESULTADOS — VFL VERTICAL\n"
        f"  {'Dirección':<12} | "
        f"{'a.VFL (°/s)':^11} | "
        f"{'Nistagmos/s':^11} | "
        f"{'Nistagmos':^9}\n"
        f"  {'─' * 51}\n"
        f"  {'Arriba':<12} | "
        f"{datos.get('avfl_up', 0.0):^11.1f} | "
        f"{datos.get('nps_up', 0.0):^11.2f} | "
        f"{datos.get('n_up', 0):^9}\n"
        f"  {'Abajo':<12} | "
        f"{datos.get('avfl_down', 0.0):^11.1f} | "
        f"{datos.get('nps_down', 0.0):^11.2f} | "
        f"{datos.get('n_down', 0):^9}"
    )

    fig.text(
        0.5,
        0.005,
        tabla_vfl,
        ha="center",
        va="bottom",
        family="monospace",
        size=9,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.94
        }
    )

    # Reserva espacio para la tabla de resultados.
    fig.subplots_adjust(
        bottom=0.30,
        top=0.92
    )

    plt.show()

