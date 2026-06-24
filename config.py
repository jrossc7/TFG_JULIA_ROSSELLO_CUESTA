"""
config.py
Parámetros globales del sistema de análisis de videonistagmografía (VNG).
"""

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS Y LISTA DE VÍDEOS
# ─────────────────────────────────────────────────────────────────────────────

VIDEOS_A_PROCESAR = [
    #"dataset/añadir nombre del video.mp4/mov",
]

# True: abre selector gráfico al arrancar. False: usa VIDEOS_A_PROCESAR.
USE_FILE_DIALOG = True


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESAMIENTO Y VISIÓN POR COMPUTADOR
# ─────────────────────────────────────────────────────────────────────────────

BORDER_CROP = 4  # [px]


# ── Seguimiento dinámico de la pupila ────────────────────────────────────────

TRACKING_WINDOW       = 180  # [px]
TRACKING_WINDOW_PANIC = 360  # [px]
MAX_REENGAGE_DIST     = 80   # [px]


# ── Filtrado gaussiano ───────────────────────────────────────────────────────

GAUSSIAN_KERNEL = (9, 9)  # tamaño impar en ambos ejes
GAUSSIAN_SIGMA  = 0       # 0 = OpenCV calcula sigma automáticamente


# ── Operaciones morfológicas ─────────────────────────────────────────────────

MORPH_CLOSE_K    = (11, 11)  # clausura para rellenar reflejos
MORPH_CLOSE_ITER = 2
OPEN_BRIDGE_K    = (9, 9)    # apertura para eliminar conexiones finas


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN DE CONTORNOS Y LOCALIZACIÓN DE LA PUPILA
# ─────────────────────────────────────────────────────────────────────────────

MIN_ELLIPSE_RATIO      = 0.35
ASPECT_MIN             = 0.60
ASPECT_MAX             = 1.50
SOLIDITY_MIN           = 0.90   # área / área envolvente convexa
MAX_PUPIL_AREA_FRACTION = 0.60  # fracción del blob más grande descartado
MAX_PUPIL_AREA_FALLBACK = 40_000  # [px²]
ROI_MARGIN_X           = 0.15
ROI_MARGIN_Y           = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRACIÓN
# ─────────────────────────────────────────────────────────────────────────────

# Con 1.0 las unidades son px y px/s en lugar de grados y °/s.
PIXELS_PER_DEGREE = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FILTRADO DE POSICIÓN Y VELOCIDAD
# ─────────────────────────────────────────────────────────────────────────────

SG_WINDOW_MS  = 700.0  # [ms] Savitzky-Golay sobre la posición
SG_POLYORD    = 3

SPV_WINDOW_MS = 366.7  # [ms] Savitzky-Golay sobre la velocidad de fase lenta
SPV_POLYORD   = 2

OUTLIER_THRESHOLD = 3.5  # umbral en unidades MAD
OUTLIER_WINDOW    = 9    # [frames]


# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DE FASES RÁPIDAS
# ─────────────────────────────────────────────────────────────────────────────

PEAK_HEIGHT   = 22.0  # [°/s] altura mínima del pico de velocidad
PEAK_HEIGHT_K = 0.0   # multiplicador MAD para umbral adaptativo (desactivado)

PEAK_PROM     = 10.0  # [°/s] prominencia mínima del pico
PEAK_PROM_K   = 0.0   # multiplicador MAD para umbral adaptativo (desactivado)


# ── Posición específica para detectar sacadas ────────────────────────────────

FAST_POS_WINDOW_MS = 233.3  # [ms]
FAST_POS_POLYORD   = 2

# ── Velocidad específica para detectar sacadas ───────────────────────────────

FAST_VEL_WINDOW_MS = 166.7  # [ms]
FAST_VEL_POLYORD   = 2
MIN_BEAT_DIST_MS   = 160.0  # [ms] separación mínima entre fases rápidas


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN Y CÁLCULO DE LA FASE LENTA
# ─────────────────────────────────────────────────────────────────────────────

SPV_LOOKBACK_MS       = 250.0  # [ms] ventana anterior a la sacada
PRE_SACCADE_GUARD_MS  = 30.0   # [ms] margen antes del inicio de la sacada
POST_SACCADE_GUARD_MS = 30.0   # [ms] margen tras la sacada anterior
SPV_NOISE_FLOOR       = 0.3    # [°/s] velocidad mínima de fase lenta válida
MIN_FAST_SLOW_RATIO   = 1.5    # ratio mínimo velocidad rápida / velocidad lenta


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

Y_AXIS_LIMIT   = 50.0
Y_AXIS_LIMIT_X = 50.0  # [°]
Y_AXIS_LIMIT_Y = 25.0  # [°]


# ─────────────────────────────────────────────────────────────────────────────
# INTERFAZ DE USUARIO
# ─────────────────────────────────────────────────────────────────────────────

MAX_W = 800  # [px]

DEBUG_PANEL_W = 280  # [px]
DEBUG_PANEL_H = 280  # [px]

# Si True, find_pupil_in_mask imprime en consola el motivo de aceptación/rechazo
# de cada contorno. Solo para diagnóstico puntual.
DEBUG_TRACKING_VERBOSE = False

MAIN_WINDOW_W = 400  # [px]
MAIN_WINDOW_H = 300  # [px]


# ── Valores iniciales de los controles ───────────────────────────────────────

DEFAULT_THRESH   = 80
DEFAULT_MIN_AREA = 100   # [px²]
DEFAULT_MIN_CIRC = 0.10  # ×100 en el trackbar → posición inicial 10


# ── Nombres de ventanas ──────────────────────────────────────────────────────

WIN_MAIN = (
    "VNG - Deteccion de Pupila  "
    "[q=salir | p=pausar | d=debug | s=guardar]"
)

WIN_DEBUG = "Debug"


# ── Nombres de trackbars ─────────────────────────────────────────────────────

TRACK_THRESH = "Umbral Pupila"
TRACK_AREA   = "Filtro Area Min"
TRACK_CIRC   = "Circularidad Min"