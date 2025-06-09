import math
from pathlib import Path
from typing import Any, Dict

from loguru import logger
from tqdm import tqdm
import typer

from PIL import Image
import matplotlib.pyplot as plt

from vision_transformer.config import DATASET_CONFIG, FIGURES_DIR, PROCESSED_DATA_DIR
from vision_transformer.utils import DatasetFormat

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


if __name__ == "__main__":
    app()
