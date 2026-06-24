# TFG: Diseño y desarrollo de un sistema de asistencia a la detección de nistagmo

Este repositorio contiene el código desarrollado como parte del **Trabajo de Fin de Grado (TFG)** de **Julia Rosselló Cuesta**, orientado al diseño e implementación de un prototipo software capaz de procesar vídeos oculares procedentes de un videonistagmoscopio para detectar y caracterizar nistagmo de forma automática.

---

## Contexto y Motivación

El **nistagmo** es un movimiento ocular involuntario y uno de los principales signos clínicos empleados en la evaluación de los trastornos del equilibrio y del sistema vestibular. Su registro se realiza habitualmente mediante sistemas de **videonistagmografía (VNG)**, equipos especializados de alto coste y disponibilidad limitada a entornos hospitalarios.

Este TFG propone una alternativa accesible: un sistema de procesamiento de imagen en Python que, sin necesidad de hardware especializado ni modelos de aprendizaje profundo, es capaz de:

- **Segmentar y rastrear la pupila** a lo largo del vídeo mediante técnicas clásicas de visión por computador (umbralización, operaciones morfológicas, análisis de contornos).
- **Calcular parámetros clínicos** asociados al nistagmo: velocidad de fase lenta (VFL) y número de batidas por dirección.
- **Generar representaciones gráficas** (nistagmograma, velocidad, marcado de sacadas) para asistir en la interpretación clínica.

Los resultados se comparan con los obtenidos por un sistema comercial de referencia (Interacoustics VisualEyes VNG 515).

---

## Estructura del Proyecto

```
VNG/
├── main.py           # Punto de entrada. Orquesta la lectura de vídeo, el tracking y el análisis
├── tracking.py       # Módulo de visión por computador: preprocesamiento y detección de la pupila
├── analisis.py       # Módulo de procesamiento matemático y clínico (VFL, detección de sacadas)
├── visualizacion.py  # Generación de las gráficas clínicas finales
└── config.py         # Parámetros globales del sistema (umbrales, filtros, ventanas, etc.)
```

---

## Requisitos

- **Python** ≥ 3.9
- Dependencias (ver `requirements.txt`):

```bash
pip install -r requirements.txt
```

Las principales librerías utilizadas son:

| Librería | Uso |
|---|---|
| `opencv-python` | Captura de vídeo, preprocesamiento y visualización en tiempo real |
| `numpy` | Operaciones matemáticas vectorizadas |
| `scipy` | Filtro Savitzky-Golay y detección de picos (`find_peaks`) |
| `matplotlib` | Generación de las gráficas clínicas finales |

---

## Uso

### 1. Selector gráfico de vídeo (modo por defecto)

Con `USE_FILE_DIALOG = True` en `config.py`, al ejecutar el programa se abrirá automáticamente un selector de archivos nativo:

```bash
python main.py
```

Se pueden seleccionar uno o varios vídeos (`.mp4`, `.mov`, `.avi`). El programa los procesará secuencialmente.

### 2. Lista de vídeos predefinida

Alternativamente, en `config.py` se puede desactivar el diálogo y definir la lista manualmente:

```python
# config.py
USE_FILE_DIALOG = False
VIDEOS_A_PROCESAR = [
    "ruta/al/video1.mp4",
    "ruta/al/video2.mov",
]
```

---

## Controles durante la ejecución

| Tecla | Acción |
|---|---|
| `q` | Salir / pasar al siguiente vídeo |
| `p` | Pausar / reanudar |
| `d` | Activar / desactivar panel de debug |
| `s` | Guardar el frame actual como imagen PNG |

Durante la pausa, los **trackbars** permiten ajustar en tiempo real el umbral de binarización, el área mínima y la circularidad mínima de la pupila.

---

## Resultados generados

Al finalizar el análisis de cada vídeo, el programa muestra en consola un **resumen de calidad del tracking** y las **métricas clínicas** (VFL media, número de nistagmos por dirección y nistagmos/segundo), y abre automáticamente las **gráficas clínicas**:

- **Nistagmograma**: posición horizontal y vertical de la pupila a lo largo del tiempo, con marcado de las fases rápidas detectadas.
- **Velocidad**: señal de velocidad angular y umbral de detección de sacadas.

---

## Parámetros de configuración (`config.py`)

Todos los hiperparámetros del sistema son ajustables en `config.py` sin modificar el código principal:

| Parámetro | Descripción |
|---|---|
| `PIXELS_PER_DEGREE` | Factor de calibración píxeles→grados (1.0 = unidades en píxeles) |
| `PEAK_HEIGHT` | Velocidad mínima (°/s) para detectar una fase rápida |
| `SG_WINDOW_MS` | Ventana del filtro Savitzky-Golay sobre la posición (ms) |
| `SPV_LOOKBACK_MS` | Ventana de cálculo de la velocidad de fase lenta (ms) |
| `TRACKING_WINDOW` | Tamaño de la ventana de seguimiento dinámico (px) |
| `Y_AXIS_LIMIT_X` / `Y_AXIS_LIMIT_Y` | Límites del eje Y en las gráficas (°) |

---

##  Notas

> [!NOTE]
> El sistema está diseñado para vídeos grabados con un **videonistagmoscopio en condiciones de oscuridad**, donde la pupila aparece como la región más oscura de la imagen. El rendimiento puede variar con vídeos grabados en otras condiciones.

> [!IMPORTANT]
> El parámetro `PIXELS_PER_DEGREE` en `config.py` debe calibrarse para cada configuración óptica concreta. Con el valor por defecto (1.0), todas las magnitudes se expresan en píxeles y píxeles/segundo en lugar de grados y °/s.
