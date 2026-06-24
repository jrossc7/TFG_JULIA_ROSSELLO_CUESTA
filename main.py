"""
main.py
Punto de entrada principal. Coordina lectura de vídeo, tracking,
análisis clínico y visualización.
"""

import cv2
import csv
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

import config
from tracking import crop_border, to_grayscale, reduce_noise, threshold_pupil, morphological_cleanup, detect_pupil
from analisis import procesar_datos_clinicos
from visualizacion import create_debug_window, read_trackbars, build_debug_panel, build_main_view, draw_detection, mostrar_graficas_clinicas


def _resumen_calidad_tracking(time_history: list, circ_history: list,
                               low_thresh: float = 0.5, min_run_frames: int = 3) -> None:
    """Imprime un resumen diagnóstico de la calidad de detección a lo largo del vídeo,
    señalando tramos sostenidos de circularidad baja (posible oclusión parcial)."""
    circ = np.array(circ_history, dtype=float)
    t = np.array(time_history, dtype=float)
    valid = np.isfinite(circ)
    if not np.any(valid):
        return

    n_nan = int(np.sum(~valid))
    print("\n[CALIDAD DE TRACKING]")
    print(f"  Frames sin detección (NaN)   : {n_nan} / {len(circ)}")
    print(f"  Circularidad media (válidos) : {np.nanmean(circ):.2f}")
    print(f"  Circularidad mínima          : {np.nanmin(circ):.2f}")

    low = valid & (circ < low_thresh)
    idx = np.where(low)[0]
    if idx.size == 0:
        print(f"  Sin tramos sostenidos de circularidad < {low_thresh:.2f}")
        return

    print(f"  Tramos con circularidad < {low_thresh:.2f} (posible oclusión parcial):")
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        if prev - start + 1 >= min_run_frames:
            print(f"    t = {t[start]:.2f}s a {t[prev]:.2f}s  ({prev - start + 1} frames)")
        start = prev = i
    if prev - start + 1 >= min_run_frames:
        print(f"    t = {t[start]:.2f}s a {t[prev]:.2f}s  ({prev - start + 1} frames)")


def _exportar_datos_debug(video_path: str, time_history: list, x_history: list,
                           y_history: list, circ_history: list) -> None:
    """Exporta los datos crudos (tiempo, posición, circularidad) a un CSV."""
    nombre = Path(video_path).stem
    out_path = Path(f"debug_{nombre}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "t_seg", "x_px", "y_px", "circularidad"])
        for i in range(len(time_history)):
            writer.writerow([
                i,
                time_history[i],
                x_history[i],
                y_history[i],
                circ_history[i] if i < len(circ_history) else "",
            ])
    print(f"[INFO] Datos crudos exportados a: {out_path.resolve()}")


def analizar_ojo(video_path: str, show_debug: bool = True) -> None:
    """Analiza un vídeo VNG completo orquestando los diferentes módulos."""
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Fichero no encontrado: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el vídeo: {video_path}")

    fps_meta   = cap.get(cv2.CAP_PROP_FPS)
    total_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Procesando: {Path(video_path).name} ({total_meta} frames)")

    cv2.namedWindow(config.WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WIN_MAIN, config.MAIN_WINDOW_W, config.MAIN_WINDOW_H)
    create_debug_window()

    try:
        frame_idx   = 0
        paused      = False
        debug_on    = show_debug

        fps_video    = fps_meta or 25.0
        time_history = []
        x_history    = []
        y_history    = []
        circ_history = []

        vis_frame    = None
        debug_panel  = None
        gray_smooth  = None
        binary_mask  = None
        crop_off     = (config.BORDER_CROP, config.BORDER_CROP)
        roi_rect     = None

        last_cx = last_cy = last_known_cx = last_known_cy = None
        lost_frames_count = 0
        frame_bgr_frozen = None

        while True:
            thresh_val, min_area, min_circ, min_circ_pct = read_trackbars()

            if not paused:
                ret, frame_bgr = cap.read()
                if not ret: break
                frame_idx += 1
                frame_bgr_frozen = frame_bgr

                h, w = frame_bgr.shape[:2]
                current_tracking_window = None

                # Selección de la región de búsqueda
                if last_cx is not None and last_cy is not None:
                    current_tracking_window = config.TRACKING_WINDOW
                elif lost_frames_count == 1 and last_known_cx is not None:
                    last_cx = last_known_cx
                    last_cy = last_known_cy
                    current_tracking_window = config.TRACKING_WINDOW_PANIC

                if current_tracking_window is not None:
                    cx_int = int(round(last_cx))
                    cy_int = int(round(last_cy))

                    # Ventana desplazada (no recortada) cerca del borde
                    win_w = min(current_tracking_window, w)
                    win_h = min(current_tracking_window, h)

                    x0 = cx_int - win_w // 2
                    x0 = max(0, min(x0, w - win_w))
                    x1 = x0 + win_w

                    y0 = cy_int - win_h // 2
                    y0 = max(0, min(y0, h - win_h))
                    y1 = y0 + win_h

                    search_frame = frame_bgr[y0:y1, x0:x1]
                    crop_off     = (x0, y0)
                    roi_rect     = (x0, y0, x1 - x0, y1 - y0)
                else:
                    search_frame = crop_border(frame_bgr, config.BORDER_CROP)
                    crop_off     = (config.BORDER_CROP, config.BORDER_CROP)
                    roi_rect     = None

                if search_frame is None or search_frame.size == 0:
                    last_cx = last_cy = None
                    lost_frames_count += 1
                    continue

                # Preprocesamiento
                gray_raw    = to_grayscale(search_frame)
                gray_smooth = reduce_noise(gray_raw)

                ITER_STEP = 5
                ITER_MIN  = 20
                current_thresh = thresh_val
                pupil_result   = None

                # Reducción iterativa del umbral si no hay detección
                while current_thresh >= ITER_MIN:
                    binary       = threshold_pupil(gray_smooth, current_thresh)
                    binary_mask  = morphological_cleanup(binary)
                    pupil_result = detect_pupil(binary_mask, min_area, min_circ, crop_off)
                    if pupil_result is not None: break
                    current_thresh -= ITER_STEP

                # Validación de reenganche por distancia
                if pupil_result is not None:
                    new_cx, new_cy = pupil_result["center"]
                    if lost_frames_count > 0 and last_known_cx is not None:
                        dist = math.hypot(new_cx - last_known_cx, new_cy - last_known_cy)
                        if dist > config.MAX_REENGAGE_DIST: pupil_result = None

                # Actualizar estado de tracking
                if pupil_result is not None:
                    last_cx = last_known_cx = pupil_result["center"][0]
                    last_cy = last_known_cy = pupil_result["center"][1]
                    lost_frames_count = 0
                else:
                    last_cx = last_cy = None
                    lost_frames_count += 1

                # Guardar datos
                t_sec = frame_idx / fps_video
                time_history.append(t_sec)
                if pupil_result is not None:
                    x_history.append(float(pupil_result["center"][0]))
                    y_history.append(float(pupil_result["center"][1]))
                    circ_history.append(float(pupil_result.get("circularity", np.nan)))
                else:
                    x_history.append(np.nan)
                    y_history.append(np.nan)
                    circ_history.append(np.nan)

                vis_frame = draw_detection(frame_bgr, pupil_result, frame_idx,
                                           thresh_val, min_area, min_circ_pct, roi_rect)

            else:
                # Modo pausa: ajuste manual de umbrales sobre el frame congelado
                if gray_smooth is not None and frame_bgr_frozen is not None:
                    binary_live  = threshold_pupil(gray_smooth, thresh_val)
                    binary_mask  = morphological_cleanup(binary_live)
                    pupil_result = detect_pupil(binary_mask, min_area, min_circ, crop_off,
                                                verbose=config.DEBUG_TRACKING_VERBOSE)
                    vis_frame    = draw_detection(frame_bgr_frozen, pupil_result, frame_idx,
                                                  thresh_val, min_area, min_circ_pct, roi_rect)

            # Mostrar ventanas
            if vis_frame is not None:
                img_show = build_main_view(vis_frame)
                cv2.imshow(config.WIN_MAIN, img_show)

            if debug_on and gray_smooth is not None:
                filled_dbg = pupil_result.get("filled_mask") if pupil_result else None
                if roi_rect is not None:
                    rx, ry, rw, rh = roi_rect
                    vis_frame_debug = vis_frame[ry:ry + rh, rx:rx + rw]
                else:
                    vis_frame_debug = vis_frame
                debug_panel = build_debug_panel(vis_frame_debug, gray_smooth,
                                                binary_mask, filled_dbg)
                h_d, w_d = debug_panel.shape[:2]
                if w_d > config.MAX_W:
                    new_h_d = int(h_d * (config.MAX_W / w_d))
                    debug_show = cv2.resize(debug_panel, (config.MAX_W, new_h_d),
                                            interpolation=cv2.INTER_AREA)
                else:
                    debug_show = debug_panel
                cv2.imshow(config.WIN_DEBUG, debug_show)

            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): paused = not paused
            elif key == ord('d'): debug_on = not debug_on
            elif key == ord('s') and vis_frame is not None:
                cv2.imwrite(f"frame_{frame_idx:05d}.png", vis_frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Análisis de vídeo finalizado.")

    if len(time_history) > 1:
       #_exportar_datos_debug(video_path, time_history, x_history, y_history, circ_history)
        _resumen_calidad_tracking(time_history, circ_history)
        datos_procesados = procesar_datos_clinicos(time_history, x_history, y_history, fps_video)
        mostrar_graficas_clinicas(datos_procesados)
    else:
        print("[INFO] No hay datos suficientes para generar las gráficas.")


# ─────────────────────────────────────────────────────────────────────────────
# EJECUCIÓN POR LOTES
# ─────────────────────────────────────────────────────────────────────────────

def seleccionar_videos() -> list:
    """Abre un selector de archivos nativo para elegir uno o varios vídeos."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    rutas = filedialog.askopenfilenames(
        title="Selecciona el/los vídeo(s) a analizar",
        filetypes=[("Vídeos", "*.mp4 *.mov *.avi"), ("Todos los archivos", "*.*")]
    )
    root.destroy()
    return list(rutas)


if __name__ == "__main__":
    if config.USE_FILE_DIALOG:
        videos_a_procesar = seleccionar_videos()
        if not videos_a_procesar:
            print("[AVISO] No se ha seleccionado ningún vídeo.")
    else:
        videos_a_procesar = config.VIDEOS_A_PROCESAR

    n_total = len(videos_a_procesar)
    if n_total == 0:
        print("[AVISO] No hay vídeos para procesar.")

    for i, video in enumerate(videos_a_procesar, start=1):
        print(f"\n► [{i}/{n_total}] Iniciando: {Path(video).name}")

        if not Path(video).exists():
            print(f"  [SKIP] Fichero no encontrado: {video}")
            continue

        try:
            analizar_ojo(video, show_debug=True)
        except Exception as exc:
            print(f"  [ERROR] Fallo al procesar {Path(video).name}: {exc}")
            cv2.destroyAllWindows()
            plt.close("all")
            continue

        print(f"  ✔ [{i}/{n_total}] Completado: {Path(video).name}")
        if i < n_total:
            siguiente = Path(videos_a_procesar[i]).name
            print(f"     → Siguiente: {siguiente}  (se iniciará al cerrar las gráficas)")

    print("\n[BATCH] Sesión completada. Todos los vídeos han sido procesados.")