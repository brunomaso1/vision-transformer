from collections import Counter
import math
from pathlib import Path
from typing import Any, Dict, List

import cv2
from PIL import Image
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import typer

import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
from pathlib import Path

from vision_transformer.config import DATASET_CONFIG, FIGURES_DIR, PROCESSED_DATA_DIR
from vision_transformer.utils import DatasetFormat

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


def show_image(image: Image, label: str = "Unknown") -> None:
    """
    Muestra una imagen y su etiqueta.
    Args:
        image (Image.Image): Imagen a mostrar.
        label (str): Etiqueta de la imagen.
    """
    plt.imshow(image)
    plt.title(f"Etiqueta: {label}")
    plt.axis("off")
    plt.show()


def show_image_grid(data: dict, title: str = "Grilla de Imágenes", num_cols: int = 4) -> None:
    """
    Muestra una grilla de imágenes con sus etiquetas.

    Args:
        data (dict): Diccionario con las claves "images" (lista de objetos Image)
                     y "labels" (lista de strings de etiquetas).
        title (str): Título general para la grilla de imágenes.
        num_cols (int): Número de columnas en la grilla.
    """
    images = data.get("images", [])
    labels = data.get("labels", [])

    if not images:
        logger.warning("No hay imágenes para mostrar.")
        return

    num_images = len(images)
    num_cols = max(1, min(num_cols, num_images))
    num_rows = math.ceil(num_images / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(images[i])
        ax.set_title(f"Etiqueta: {labels[i] if i < len(labels) else 'Unknown'}")
        ax.axis("off")

    # Ocultar subplots vacíos si hay...
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_class_distribution(dataset, id2label, split="train"):
    """
    Grafica la distribución de clases.
    """
    counts = Counter(dataset[split]["label"])

    df = pd.DataFrame([{"class": f"{idx}: {name}", "count": counts.get(idx, 0)} for idx, name in id2label.items()])
    df = df.sort_values(by=["count", "class"], ascending=[True, False])

    ax = df.plot.barh(x="class", y="count", figsize=(6, 4), legend=False)
    ax.set_title(f"Cantidad de imágenes por clase en el set de {split}", fontsize=10)
    ax.set_ylabel("")

    for i, (count) in enumerate(df["count"]):
        ax.text(
            count - max(df["count"]) * 0.01,
            i,
            str(count),
            va="center",
            ha="right",
            color="white",
            fontweight="bold",
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


def _compute_class_histograms(dataset, id2label, split="train"):
    class_histograms = {}

    y = np.array(dataset[split]["label"])
    for label_id, label_name in id2label.items():
        imgs_idx = np.where(y == label_id)[0]

        # Accumulate histograms per channel
        hist_H = np.zeros(180, dtype=np.float32)
        hist_S = np.zeros(256, dtype=np.float32)
        hist_L = np.zeros(256, dtype=np.float32)

        for idx in imgs_idx:
            img_pil = dataset[split][int(idx)]["image"]
            img_hls = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2HLS)

            # Split H, L, S channels
            H, L, S = cv2.split(img_hls)

            hist_H += cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
            hist_S += cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
            hist_L += cv2.calcHist([L], [0], None, [256], [0, 256]).flatten()

        class_histograms[label_name] = [hist_H, hist_S, hist_L]

    return class_histograms


def _get_colormap_bar_colors(channel, bins=256):
    """Return a list of RGB colors representing the channel values visually."""
    if channel == "H":  # Hue (0–179 in OpenCV HLS, but assuming 0–255 here)
        hsv = np.stack([np.linspace(0, 179, bins), np.full(bins, 255), np.full(bins, 255)], axis=1).astype(np.uint8)
        rgb = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)
    elif channel == "S":  # Saturation
        hsv = np.stack([np.full(bins, 0), np.linspace(0, 255, bins), np.full(bins, 200)], axis=1).astype(np.uint8)
        rgb = cv2.cvtColor(hsv.reshape(-1, 1, 3), cv2.COLOR_HSV2RGB).reshape(-1, 3)
    elif channel == "L":  # Lightness
        rgb = np.stack([np.linspace(0, 255, bins)] * 3, axis=1)
    else:
        raise ValueError(f"Invalid channel: {channel}")
    return rgb / 255  # Normalize for matplotlib


def plot_class_histograms(dataset, id2label, split="train"):
    class_histograms = _compute_class_histograms(dataset, id2label, split)

    channel_bins = {"H": 180, "S": 256, "L": 256}

    class_names = list(class_histograms.keys())
    channel_names = ["H", "S", "L"]
    full_titles = ["Hue", "Saturation", "Lightness"]

    num_classes = len(class_names)
    fig, axes = plt.subplots(num_classes, 3, figsize=(9, num_classes * 1), sharex=True, sharey=False)

    if num_classes == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, class_name in enumerate(class_names):
        histograms = class_histograms[class_name]

        for col_idx, (channel, hist) in enumerate(zip(channel_names, histograms)):
            ax = axes[row_idx][col_idx]

            bins = channel_bins[channel]
            colors = _get_colormap_bar_colors(channel, bins)

            # Normalize histogram
            hist = hist[:bins]  # Ensure it's trimmed properly
            hist = hist / hist.max()

            for x in range(bins):
                ax.bar(x, hist[x], color=colors[x], width=1)

            ax.set_xlim([0, bins])
            ax.set_ylim([0, 1])
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(full_titles[col_idx], fontsize=10)

            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=9, rotation=0, labelpad=55, va="center")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    fig.suptitle("Histograma acumulado por clase para los canales HSL", fontsize=16)
    plt.show()


def plot_metric(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str,
    filename: str,
    dirpath: Path = FIGURES_DIR,
    y_labels: List[str] = None,
) -> None:
    """
    Grafica métricas en función de una columna específica del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos a graficar.
        x_col (str): Nombre de la columna que se usará como eje X.
        y_cols (List[str]): Lista de nombres de columnas que se graficarán en el eje Y.
        title (str): Título del gráfico.
        filename (str): Nombre del archivo para guardar el gráfico.
        dirpath (Path, optional): Directorio donde se guardará el gráfico. Por defecto es FIGURES_DIR.
        y_labels (List[str], optional): Lista de etiquetas para las métricas en el gráfico. Por defecto es None.

    Ejemplo de uso:
        >>> df = pd.DataFrame({
        ...    "epoch": [1, 2, 3, 4],
        ...    "accuracy": [0.8, 0.85, 0.9, 0.92],
        ...    "loss": [0.5, 0.4, 0.3, 0.2]
        ... })
        >>> plot_metric(
        ...    df=df,
        ...    x_col="epoch",
        ...    y_cols=["accuracy", "loss"],
        ...    title="Curvas de Métricas",
        ...    filename="metric_curves",
        ...    y_labels=["Precisión", "Pérdida"]
        ... )
    """
    fig = go.Figure()

    for i, y_col in enumerate(y_cols):
        fig.add_trace(
            go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers", name=y_labels[i] if y_labels else y_col)
        )

    fig.update_layout(
        title=title, xaxis_title=x_col, yaxis_title="Valor", legend_title="Métrica", template="plotly_white"
    )

    fig.show()

    dirpath.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(dirpath / f"{filename}.html"))
    fig.write_image(str(dirpath / f"{filename}.png"))


def plot_confusion_matrix(
    y_true, y_pred, filename, dirpath: Path = FIGURES_DIR, labels=None, show_as_percentaje=True
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype(np.int32)

    # Calcular porcentajes por fila
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, cm_sum, where=cm_sum != 0) * 100

    # Combinar texto: valor (xx) + porcentaje (yy%)
    z_text = (
        [
            [
                f"{pct:.1f}%" if cm_sum[i][0] != 0 else "0"
                for j, (val, pct) in enumerate(zip(row, cm_percent[i]))
            ]
            for i, row in enumerate(cm)
        ]
        if show_as_percentaje
        else cm.astype(str).tolist()
    )

    fig = ff.create_annotated_heatmap(
        z=cm, x=labels, y=labels, annotation_text=z_text, colorscale="Blues", showscale=True, reversescale=False
    )

    fig.update_layout(
        title="Matriz de Confusion",
        xaxis_title="Etiqueta Predicha",
        yaxis_title="Etiqueta Verdadera",
        template="plotly_white",
    )

    fig.update_yaxes(autorange="reversed")

    fig.show()

    dirpath.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(dirpath / f"{filename}.html"))
    fig.write_image(str(dirpath / f"{filename}.png"))


if __name__ == "__main__":
    app()
