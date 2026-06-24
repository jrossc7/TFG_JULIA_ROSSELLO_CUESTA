"""
tracking.py
Módulo de Visión por Computador.
Se encarga del preprocesamiento de la imagen y la detección del centroide de la pupila.
"""

import cv2
import numpy as np
import math
import config

# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 2 — PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def crop_border(frame_bgr: np.ndarray, margin: int) -> np.ndarray:
    """Elimina un margen uniforme en cada borde del frame."""
    if margin <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    return frame_bgr[margin:h - margin, margin:w - margin]

def to_grayscale(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

def reduce_noise(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, config.GAUSSIAN_KERNEL, config.GAUSSIAN_SIGMA)

def threshold_pupil(gray: np.ndarray, thresh_val: int) -> np.ndarray:
    _, binary = cv2.threshold(
        gray, thresh_val, 255, cv2.THRESH_BINARY_INV
    )
    return binary

def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_CLOSE_K)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                            iterations=config.MORPH_CLOSE_ITER)

def preprocess_frame(frame_bgr: np.ndarray, thresh_val: int) -> tuple:
    """
    Cadena completa de preprocesamiento
    BGR → Gris → Gaussiano → Umbral Manual → Clausura
    """
    gray_raw    = to_grayscale(frame_bgr)
    gray_smooth = reduce_noise(gray_raw)
    binary      = threshold_pupil(gray_smooth, thresh_val)
    mask        = morphological_cleanup(binary)
    
    return gray_raw, gray_smooth, mask

# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO 3 — LOCALIZACIÓN DE LA PUPILA
# ─────────────────────────────────────────────────────────────────────────────

def circularity(contour) -> float:
    """Calcula la circularidad de un contorno: 4π·A / P²"""
    area = cv2.contourArea(contour)
    if area == 0:
        return 0.0
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter ** 2)

def find_pupil_in_mask(mask: np.ndarray,
                       min_area: int,
                       min_circ: float,
                       roi_offset: tuple = (0, 0),
                       verbose: bool = False):
    """Localiza la pupila en una máscara binaria mediante filtros en cadena.
    verbose=True activa el log de diagnóstico por contorno (ver DEBUG_TRACKING_VERBOSE)."""

    def _v(msg):
        if verbose:
            print(f"    [find_pupil] {msg}")

    # ── Paso 0: Apertura morfológica anti-puente ──────────────────────────────────────
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.OPEN_BRIDGE_K)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # ── Paso 0b: Máscara de Zona Segura (Safe-Zone ROI) ──────────────────
    h_m, w_m = mask_opened.shape[:2]
    x0 = int(w_m * config.ROI_MARGIN_X)
    x1 = int(w_m * (1.0 - config.ROI_MARGIN_X))
    y0 = int(h_m * config.ROI_MARGIN_Y)
    y1 = int(h_m * (1.0 - config.ROI_MARGIN_Y))
    safe_mask = np.zeros_like(mask_opened)
    safe_mask[y0:y1, x0:x1] = mask_opened[y0:y1, x0:x1]

    # ── Paso 0c: Cortafuegos del borde (Border Clearing) ─────────────────
    cv2.rectangle(safe_mask, (0, 0), (w_m - 1, h_m - 1), 0, thickness=3)

    contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        _v(f"Sin contornos tras zona segura (ROI_MARGIN_X={config.ROI_MARGIN_X}, "
           f"ROI_MARGIN_Y={config.ROI_MARGIN_Y}). Si la pupila estaba cerca del "
           f"borde del recorte, puede haber quedado fuera de la zona segura.")
        return None

    # ── Paso 1: Descarte dinámico del blob más grande (anti-pestaña) ─────
    areas = [(cv2.contourArea(c), c) for c in contours]
    areas.sort(key=lambda x: x[0], reverse=True)
    _v(f"{len(areas)} contorno(s) tras zona segura. Áreas (desc.): "
       f"{[round(a, 1) for a, _ in areas]}")

    if len(areas) >= 2:
        largest_area, _ = areas[0]
        dynamic_max_area = min(largest_area * config.MAX_PUPIL_AREA_FRACTION,
                               config.MAX_PUPIL_AREA_FALLBACK)
        _v(f"Contorno más grande (área={largest_area:.1f}) DESCARTADO por "
           f"heurística anti-pestaña. dynamic_max_area={dynamic_max_area:.1f} "
           f"para el resto de candidatos.")
        candidate_contours = [c for a, c in areas[1:]
                              if min_area < a < dynamic_max_area]
    else:
        dynamic_max_area = config.MAX_PUPIL_AREA_FALLBACK
        candidate_contours = [c for a, c in areas
                              if min_area < a < dynamic_max_area]

    area_filtered = candidate_contours
    if not area_filtered:
        _v(f"Ningún contorno superviviente cumple min_area={min_area} < área < "
           f"dynamic_max_area={dynamic_max_area:.1f}.")
        return None

    # Filtro 1b: Cortafuegos Biológico (Aspect Ratio)
    aspect_filtered = []
    for c in area_filtered:
        _, _, bw, bh = cv2.boundingRect(c)
        ar = float(bw) / bh if bh > 0 else 0.0
        if config.ASPECT_MIN <= ar <= config.ASPECT_MAX:
            aspect_filtered.append(c)
        else:
            _v(f"Contorno área={cv2.contourArea(c):.1f} RECHAZADO por aspect_ratio="
               f"{ar:.2f} (rango admitido [{config.ASPECT_MIN}, {config.ASPECT_MAX}]).")
    if not aspect_filtered:
        return None

    # Filtro 1c: Solidez (Solidity)
    solid_filtered = []
    for c in aspect_filtered:
        area_c = cv2.contourArea(c)
        hull   = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = (float(area_c) / hull_area) if hull_area > 0 else 0.0
        if hull_area > 0 and solidity >= config.SOLIDITY_MIN:
            solid_filtered.append(c)
        else:
            _v(f"Contorno área={area_c:.1f} RECHAZADO por solidez={solidity:.3f} "
               f"(mínimo exigido SOLIDITY_MIN={config.SOLIDITY_MIN}).")
    if not solid_filtered:
        _v("Ningún contorno superó el filtro de solidez.")
        return None

    # Filtro 2: circularidad y relación de aspecto (Bounding Box Ratio)
    circ_filtered = []
    for c in solid_filtered:
        circ = circularity(c)
        if circ >= min_circ:
            circ_filtered.append((c, circ))
        else:
            _, _, bw, bh = cv2.boundingRect(c)
            ar = float(bw) / bh if bh > 0 else 0.0
            if 0.5 <= ar <= 1.5:
                circ_filtered.append((c, circ))
            else:
                _v(f"Contorno área={cv2.contourArea(c):.1f} RECHAZADO: circularidad="
                   f"{circ:.3f} < min_circ={min_circ:.2f} y aspect_ratio={ar:.2f} "
                   f"fuera de [0.5, 1.5] (bypass no aplicable).")

    if not circ_filtered:
        return None

    best_contour, best_circ = max(circ_filtered, key=lambda x: cv2.contourArea(x[0]))
    _v(f"Contorno ACEPTADO: área={cv2.contourArea(best_contour):.1f}, "
       f"circularidad={best_circ:.3f}.")

    # ── Relleno de agujeros (Hole Filling) ───────────────────────────────
    h_mask, w_mask = mask.shape[:2]
    filled_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
    cv2.drawContours(filled_mask, [best_contour], 0, 255, cv2.FILLED)

    # ── Centro con Image Moments sobre la máscara rellena ────────────────
    M = cv2.moments(filled_mask)
    if M['m00'] == 0:
        return None
    cx_roi = M['m10'] / M['m00']
    cy_roi = M['m01'] / M['m00']

    ox, oy = roi_offset

    # Precisión subpíxel nativa: no se trunca a entero para evitar ruido de cuantización.
    cx_global = cx_roi + ox
    cy_global = cy_roi + oy

    # ── Elipse opcional para visualización ───────────────────────────────
    ellipse_global = None
    if len(best_contour) >= 5:
        ellipse = cv2.fitEllipse(best_contour)
        (ex, ey), (minor_ax, major_ax), angle = ellipse
        if major_ax > 0 and (minor_ax / major_ax) >= config.MIN_ELLIPSE_RATIO:
            ellipse_global = ((ex + ox, ey + oy), (minor_ax, major_ax), angle)

    return {
        "center"          : (cx_global, cy_global),
        "area"            : cv2.contourArea(best_contour),
        "circularity"     : best_circ,
        "ellipse"         : ellipse_global,
        "filled_mask"     : filled_mask,
        "dynamic_max_area": dynamic_max_area,
    }

def detect_pupil(mask_full: np.ndarray,
                 min_area: int,
                 min_circ: float,
                 crop_offset: tuple = (0, 0),
                 verbose: bool = False):
    """Detecta la pupila directamente en la máscara completa."""
    return find_pupil_in_mask(mask_full, min_area, min_circ, roi_offset=crop_offset, verbose=verbose)