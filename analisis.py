"""
analisis.py
Módulo de procesamiento matemático y clínico.
"""

import numpy as np
from scipy.signal import find_peaks
import config

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES MATEMÁTICAS
# ─────────────────────────────────────────────────────────────────────────────

def _interpolate_nan(arr: np.ndarray) -> np.ndarray:
    """Interpola los valores NaN causados por fotogramas perdidos."""
    out  = arr.copy()
    nans = np.isnan(out)
    if nans.all(): return out
    idx  = np.arange(len(out))
    out[nans] = np.interp(idx[nans], idx[~nans], out[~nans])
    return out

def _remove_outliers(arr: np.ndarray, threshold: float, win: int = 9) -> np.ndarray:
    """Elimina picos usando un Z-score robusto (MAD) calculado en ventana local."""
    out, n, half = arr.copy(), len(arr), win // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        local = np.delete(out[lo:hi], i - lo)
        med = np.nanmedian(local)
        mad = np.nanmedian(np.abs(local - med)) * 1.4826
        if mad > 0 and abs(out[i] - med) / mad > threshold:
            out[i] = med
    return out

def _safe_window(n, w, p):
    """Asegura que la ventana de Savitzky-Golay sea matemáticamente válida."""
    w = min(w, n if n % 2 == 1 else n - 1)
    return w if w > p else p + 1 + (1 - (p + 1) % 2)

def _ms_to_odd_frames(ms: float, fps: float, min_frames: int = 3) -> int:
    """Convierte milisegundos al número de frames impar más cercano."""
    n = max(min_frames, int(round(fps * ms / 1000.0)))
    return n if n % 2 == 1 else n + 1

def _segment_has_gap(lost_mask: np.ndarray, a: int, b: int) -> bool:
    """True si algún frame entre a y b fue originalmente NaN antes de interpolar."""
    a, b = max(0, a), min(len(lost_mask) - 1, b)
    return bool(np.any(lost_mask[a:b + 1])) if a <= b else False

def _umbral_adaptativo(vel: np.ndarray, k: float, piso: float) -> float:
    """Umbral robusto basado en el MAD de la señal, sin bajar del piso mínimo."""
    finite = vel[np.isfinite(vel)]
    if finite.size == 0:
        return piso
    med = np.median(finite)
    mad = np.median(np.abs(finite - med)) * 1.4826
    return max(piso, k * mad)

def _savgol_numpy(arr, window, polyorder):
    """Implementación del filtro Savitzky-Golay usando NumPy."""
    half   = window // 2
    out    = np.empty_like(arr, dtype=float)
    padded = np.pad(arr, half, mode="edge")
    idx    = np.arange(window) - half
    for i in range(len(arr)):
        segment = padded[i: i + window]
        coeffs  = np.polyfit(idx, segment, polyorder)
        out[i]  = np.polyval(coeffs, 0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR PRINCIPAL DE ANÁLISIS
# ─────────────────────────────────────────────────────────────────────────────

def _detectar_nistagmo_eje(t_arr: np.ndarray,
                            clean_deg: np.ndarray,
                            was_lost: np.ndarray,
                            fps_video: float,
                            eje_nombre: str,
                            label_pos: str,
                            label_neg: str) -> dict:
    """
    Detecta sacadas y calcula la VFL sobre una señal de posición en grados.
    Reutilizable para cualquier eje; se llama una vez por eje desde
    procesar_datos_clinicos.
    """

    # ── Señal específica para detectar fases rápidas ─────────────────────────
    fast_pos_window = _ms_to_odd_frames(
        float(getattr(config, "FAST_POS_WINDOW_MS", 233.3)), fps_video)
    fast_pos_polyord = int(getattr(config, "FAST_POS_POLYORD", 2))
    fast_pos_win = _safe_window(len(clean_deg), fast_pos_window, fast_pos_polyord)
    fast_deg = _savgol_numpy(clean_deg, fast_pos_win, fast_pos_polyord)

    vel_fast_raw = np.gradient(fast_deg, t_arr)

    fast_vel_window = _ms_to_odd_frames(
        float(getattr(config, "FAST_VEL_WINDOW_MS", 166.7)), fps_video)
    fast_vel_polyord = int(getattr(config, "FAST_VEL_POLYORD", 2))
    fast_vel_win = _safe_window(len(vel_fast_raw), fast_vel_window, fast_vel_polyord)
    vel_fast = _savgol_numpy(vel_fast_raw, fast_vel_win, fast_vel_polyord)

    # ── Parámetros de detección ──────────────────────────────────────────────
    fast_height = _umbral_adaptativo(
        vel_fast, k=float(getattr(config, "PEAK_HEIGHT_K", 0.0)),
        piso=float(config.PEAK_HEIGHT))
    fast_prominence = _umbral_adaptativo(
        vel_fast, k=float(getattr(config, "PEAK_PROM_K", 0.0)),
        piso=float(config.PEAK_PROM))

    min_beat_dist_ms = float(getattr(config, "MIN_BEAT_DIST_MS", 160.0))
    dist_segura = max(2, int(round(fps_video * min_beat_dist_ms / 1000.0)))

    # ── Detección global ─────────────────────────────────────────────────────
    speed_abs = np.abs(vel_fast)
    peaks_all, _ = find_peaks(
        speed_abs, height=fast_height,
        prominence=fast_prominence, distance=dist_segura)

    # ── Parámetros de la fase lenta ──────────────────────────────────────────
    spv_lookback_ms      = float(getattr(config, "SPV_LOOKBACK_MS", 250.0))
    pre_saccade_guard_ms  = float(getattr(config, "PRE_SACCADE_GUARD_MS", 30.0))
    post_saccade_guard_ms = float(getattr(config, "POST_SACCADE_GUARD_MS", 30.0))
    noise_floor          = float(getattr(config, "SPV_NOISE_FLOOR", 0.3))
    min_fast_slow_ratio  = float(getattr(config, "MIN_FAST_SLOW_RATIO", 1.5))

    win_vfl          = max(3, int(round(fps_video * spv_lookback_ms / 1000.0)))
    pre_guard_frames  = max(1, int(round(fps_video * pre_saccade_guard_ms / 1000.0)))
    post_guard_frames = max(1, int(round(fps_video * post_saccade_guard_ms / 1000.0)))

    accepted_beats = []
    previous_candidate_peak = None
    motivos_rechazo = {}

    def _rechazar(motivo: str) -> None:
        motivos_rechazo[motivo] = motivos_rechazo.get(motivo, 0) + 1

    # ── Localización del comienzo de la fase rápida ──────────────────────────
    def find_saccade_onset(peak_idx: int) -> int:
        peak_velocity = float(vel_fast[peak_idx])
        peak_speed = abs(peak_velocity)
        if peak_velocity == 0:
            return int(peak_idx)
        peak_sign = np.sign(peak_velocity)
        onset_level = max(0.40 * fast_height, 0.20 * peak_speed)
        onset = int(peak_idx)
        while onset > 1:
            previous_velocity = float(vel_fast[onset - 1])
            if np.sign(previous_velocity) != peak_sign:
                break
            if abs(previous_velocity) < onset_level:
                break
            onset -= 1
        return onset

    # ── Validación de cada posible nistagmo ──────────────────────────────────
    for peak_idx in peaks_all:
        peak_idx = int(peak_idx)
        peak_velocity = float(vel_fast[peak_idx])
        peak_speed = abs(peak_velocity)

        if not np.isfinite(peak_velocity):
            _rechazar("velocidad no finita"); continue
        if peak_velocity == 0:
            _rechazar("velocidad exactamente cero"); continue

        direction = label_pos if peak_velocity > 0 else label_neg
        onset = find_saccade_onset(peak_idx)
        punto_B = onset - pre_guard_frames

        if punto_B < 2:
            _rechazar("inicio de sacada demasiado cerca del comienzo"); continue

        punto_A = max(0, punto_B - win_vfl)
        if previous_candidate_peak is not None:
            punto_A = max(punto_A, previous_candidate_peak + post_guard_frames)

        if punto_B - punto_A < 2:
            _rechazar("ventana de fase lenta demasiado corta (pegada al latido anterior)"); continue

        if _segment_has_gap(was_lost, punto_A, peak_idx + post_guard_frames):
            _rechazar("solapa un hueco de tracking interpolado"); continue

        t_segment = np.asarray(t_arr[punto_A:punto_B + 1], dtype=float)
        x_segment = np.asarray(clean_deg[punto_A:punto_B + 1], dtype=float)
        valid_mask = np.isfinite(t_segment) & np.isfinite(x_segment)

        if np.count_nonzero(valid_mask) < 3:
            _rechazar("muy pocas muestras válidas en la ventana de fase lenta"); continue

        t_fit, x_fit = t_segment[valid_mask], x_segment[valid_mask]
        if t_fit[-1] <= t_fit[0]:
            _rechazar("muestras de tiempo no crecientes en la ventana"); continue

        spv = float(np.polyfit(t_fit, x_fit, deg=1)[0])
        if not np.isfinite(spv):
            _rechazar("pendiente de fase lenta no finita"); continue

        spv_abs = abs(spv)
        if spv_abs <= noise_floor:
            _rechazar("VFL por debajo del suelo de ruido (SPV_NOISE_FLOOR)"); continue
        if peak_velocity < 0 and spv <= noise_floor:
            _rechazar("VFL no va en dirección opuesta a la fase rápida (peak<0)"); continue
        if peak_velocity > 0 and spv >= -noise_floor:
            _rechazar("VFL no va en dirección opuesta a la fase rápida (peak>0)"); continue

        fast_slow_ratio = peak_speed / max(spv_abs, 1e-6)
        if fast_slow_ratio < min_fast_slow_ratio:
            _rechazar("relación fase rápida/lenta insuficiente (MIN_FAST_SLOW_RATIO)"); continue

        accepted_beats.append({
            "idx": onset, "peak_idx": peak_idx, "direction": direction,
            "spv_dps": spv_abs, "fpv_dps": peak_speed,
            "fast_slow_ratio": fast_slow_ratio,
            "slow_idx_a": punto_A, "slow_idx_b": punto_B,
        })
        previous_candidate_peak = peak_idx

    # ── Separar beats por dirección ──────────────────────────────────────────
    pos_beats = [b for b in accepted_beats if b["direction"] == label_pos]
    neg_beats = [b for b in accepted_beats if b["direction"] == label_neg]

    n_pos, n_neg = len(pos_beats), len(neg_beats)
    idx_pos = np.asarray([b["idx"] for b in pos_beats], dtype=int)
    idx_neg = np.asarray([b["idx"] for b in neg_beats], dtype=int)

    avfl_pos = float(np.mean([b["spv_dps"] for b in pos_beats])) if pos_beats else 0.0
    avfl_neg = float(np.mean([b["spv_dps"] for b in neg_beats])) if neg_beats else 0.0

    t_total = float(t_arr[-1] - t_arr[0]) if len(t_arr) > 1 else 0.0
    nps_pos = n_pos / t_total if t_total > 0 else 0.0
    nps_neg = n_neg / t_total if t_total > 0 else 0.0

    print(f"\n[DETECCIÓN DE FASES RÁPIDAS - {eje_nombre}]")
    print(f"  Candidatos globales      : {len(peaks_all)}")
    print(f"  Nistagmos aceptados      : {len(accepted_beats)}")
    print(f"  {label_neg.capitalize():<12}: {n_neg}")
    print(f"  {label_pos.capitalize():<12}: {n_pos}")
    print(f"  Distancia mínima         : {dist_segura} frames ({min_beat_dist_ms:.0f} ms)")
    print(f"  Umbral / prominencia     : {fast_height:.1f} / {fast_prominence:.1f} °/s")
    if motivos_rechazo:
        print("  Motivos de rechazo de candidatos:")
        for motivo, n in sorted(motivos_rechazo.items(), key=lambda kv: -kv[1]):
            print(f"    {motivo:<60s}: {n}")
    print(f"\n[RESULTADOS CLÍNICOS - VFL {eje_nombre}]")
    print(f"  {'Dirección':<12} | {'a.VFL (°/s)':^11} | {'Nistagmos/s':^11} | {'Nistagmos':^9}")
    print(f"  {'-' * 51}")
    print(f"  {label_neg.capitalize():<12} | {avfl_neg:^11.1f} | {nps_neg:^11.2f} | {n_neg:^9}")
    print(f"  {label_pos.capitalize():<12} | {avfl_pos:^11.1f} | {nps_pos:^11.2f} | {n_pos:^9}")

    return {
        "avfl_pos": avfl_pos, "avfl_neg": avfl_neg,
        "nps_pos":  nps_pos,  "nps_neg":  nps_neg,
        "n_pos":    n_pos,    "n_neg":    n_neg,
        "idx_pos":  idx_pos,  "idx_neg":  idx_neg,
        "n_candidates": len(peaks_all), "n_accepted": len(accepted_beats),
        "motivos_rechazo": motivos_rechazo,
    }


def procesar_datos_clinicos(time_history: list,
                             x_history: list,
                             y_history: list,
                             fps_video: float) -> dict:
    """Recibe las coordenadas en bruto del vídeo y devuelve todas las métricas procesadas."""
    t_arr = np.array(time_history, dtype=float)
    x_raw = np.array(x_history,   dtype=float)
    y_raw = np.array(y_history,   dtype=float)

    # 1. Interpolación y eliminación de outliers
    x_was_lost = np.isnan(x_raw)
    y_was_lost = np.isnan(y_raw)
    outlier_win = int(getattr(config, "OUTLIER_WINDOW", 9))
    x_interp = _interpolate_nan(x_raw)
    y_interp = _interpolate_nan(y_raw)
    x_clean  = _remove_outliers(x_interp, config.OUTLIER_THRESHOLD, win=outlier_win)
    y_clean  = _remove_outliers(y_interp, config.OUTLIER_THRESHOLD, win=outlier_win)

    # 2. Suavizado de posición (Savitzky-Golay)
    sg_window_ms = float(getattr(config, "SG_WINDOW_MS", 700.0))
    sg_window_frames = _ms_to_odd_frames(sg_window_ms, fps_video)
    win  = _safe_window(len(x_clean), sg_window_frames, config.SG_POLYORD)
    x_sg = _savgol_numpy(x_clean, win, config.SG_POLYORD)
    y_sg = _savgol_numpy(y_clean, win, config.SG_POLYORD)

    # 3. Velocidad
    vel_x_raw = np.gradient(x_sg, t_arr)
    vel_y_raw = np.gradient(y_sg, t_arr)

    # 4. Suavizado de velocidad
    spv_window_ms = float(getattr(config, "SPV_WINDOW_MS", 366.7))
    spv_window_frames = _ms_to_odd_frames(spv_window_ms, fps_video)
    spv_win = _safe_window(len(vel_x_raw), spv_window_frames, config.SPV_POLYORD)
    vel_x   = _savgol_numpy(vel_x_raw, spv_win, config.SPV_POLYORD)
    vel_y   = _savgol_numpy(vel_y_raw, spv_win, config.SPV_POLYORD)

    print(f"       SPV Savitzky-Golay: ventana={spv_win} frames, orden={config.SPV_POLYORD}")

    # 5. Anclaje al cero inicial
    valid_x = x_sg[(x_sg != 0) & (~np.isnan(x_sg))]
    valid_y = y_sg[(y_sg != 0) & (~np.isnan(y_sg))]
    x_start = np.nanmean(valid_x[:5]) if len(valid_x) >= 5 else np.nanmean(x_sg)
    y_start = np.nanmean(valid_y[:5]) if len(valid_y) >= 5 else np.nanmean(y_sg)

    # 6. Conversión a grados
    x_raw_deg = -(x_raw - x_start) / config.PIXELS_PER_DEGREE
    x_sg_deg  = -(x_sg  - x_start) / config.PIXELS_PER_DEGREE
    y_raw_deg = -(y_raw - y_start) / config.PIXELS_PER_DEGREE
    y_sg_deg  = -(y_sg  - y_start) / config.PIXELS_PER_DEGREE

    vel_x_raw_deg = -vel_x_raw / config.PIXELS_PER_DEGREE
    vel_x_deg     = -vel_x     / config.PIXELS_PER_DEGREE
    vel_y_raw_deg = -vel_y_raw / config.PIXELS_PER_DEGREE
    vel_y_deg     = -vel_y     / config.PIXELS_PER_DEGREE

    x_clean_deg = -(x_clean - x_start) / config.PIXELS_PER_DEGREE
    y_clean_deg = -(y_clean - y_start) / config.PIXELS_PER_DEGREE

    # 7-8. Detección de nistagmo por eje
    res_x = _detectar_nistagmo_eje(
        t_arr, x_clean_deg, x_was_lost, fps_video,
        eje_nombre="HORIZONTAL", label_pos="right", label_neg="left")

    res_y = _detectar_nistagmo_eje(
        t_arr, y_clean_deg, y_was_lost, fps_video,
        eje_nombre="VERTICAL", label_pos="up", label_neg="down")

    return {
        "t_arr":           t_arr,
        "x_raw_deg":       x_raw_deg,     "y_raw_deg":       y_raw_deg,
        "x_sg_deg":        x_sg_deg,      "y_sg_deg":        y_sg_deg,
        "vel_x_raw_deg":   vel_x_raw_deg, "vel_y_raw_deg":   vel_y_raw_deg,
        "vel_x_deg":       vel_x_deg,     "vel_y_deg":       vel_y_deg,
        "avfl_left":       res_x["avfl_neg"], "avfl_right":   res_x["avfl_pos"],
        "nps_left":        res_x["nps_neg"],  "nps_right":    res_x["nps_pos"],
        "n_left":          res_x["n_neg"],    "n_right":      res_x["n_pos"],
        "idx_left":        res_x["idx_neg"],  "idx_right":    res_x["idx_pos"],
        "avfl_up":         res_y["avfl_pos"], "avfl_down":    res_y["avfl_neg"],
        "nps_up":          res_y["nps_pos"],  "nps_down":     res_y["nps_neg"],
        "n_up":            res_y["n_pos"],    "n_down":       res_y["n_neg"],
        "idx_up":          res_y["idx_pos"],  "idx_down":     res_y["idx_neg"],
    }