import zipfile, requests, shutil
from pathlib import Path
from typing import Optional

from loguru import logger
from tqdm import tqdm
import typer

from vision_transformer.config import (
    DATA_RAW_FILE_ZIP,
    DATA_RAW_URL,
    DATASET_CONFIG,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_DATA_EXTRACTION_DIR,
)
from vision_transformer.utils import DatasetFormat

app = typer.Typer()


@app.command()
def download_raw_data(url: str = DATA_RAW_URL, chunk_size: int = 65_536):
    """
    Descarga el archivo de datos en bruto desde la URL especificada y lo guarda en el directorio RAW_DATA_DIR.
    Muestra una barra de progreso durante la descarga.

    Args:
        url (str, optional): URL desde donde descargar el archivo de datos. Por defecto es DATA_RAW_URL.
        chunk_size (int, optional): Tamaño de cada chunk de descarga en bytes. Por defecto es 65_536.

    Raises:
        typer.Exit: Si ocurre un error durante la descarga o al guardar el archivo.

    Ejecución de ejemplo:
        >>> python -m vision_transformer.dataset download_raw_data --help
    """
    logger.info(f"Downloading raw data from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))  # Get total size of the file, if available
        filename = RAW_DATA_DIR / DATA_RAW_FILE_ZIP
        with open(filename, "wb") as file:
            # Configure tqdm progress bar if total size is known
            if total_size > 0:
                total_chunks = total_size // chunk_size + (1 if total_size % chunk_size else 0)
                logger.debug(f"Total size: {total_size} bytes, Total chunks: {total_chunks}")

                progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading")
            else:
                progress_bar = tqdm(unit="KB", desc="Downloading (unknown size)")

            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filtrar chunks vacíos
                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress_bar.update(len(chunk))
                    else:
                        progress_bar.update(len(chunk) // 1024)  # Mostrar en KB
            progress_bar.close()

        logger.success(f"Download complete! File size: {downloaded / (1024*1024):.2f} MB")
        logger.info(f"Data saved to: {filename}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading data: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def extract_raw_data(zip_file_path: Path = RAW_DATA_DIR / DATA_RAW_FILE_ZIP, clean: bool = True):
    """
    Extrae el archivo ZIP de datos en bruto y lo guarda en el directorio RAW_DATA_DIR.
    Si clean es True, elimina el archivo ZIP después de la extracción.

    Args:
        zip_file_path (Path, optional): Ruta al archivo ZIP a extraer. Por defecto es RAW_DATA_DIR / DATA_RAW_FILE.
        clean (bool, optional): Si es True, elimina el archivo ZIP después de la extracción. Por defecto es True.

    Raises:
        typer.Exit: Si ocurre un error durante la extracción o al eliminar el archivo ZIP.

    Ejecución de ejemplo:
        >>> python -m vision_transformer.dataset extract_raw_data --help
    """
    logger.info(f"Extracting raw data from {zip_file_path}...")
    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            members = zip_ref.infolist()
            with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
                for member in members:
                    zip_ref.extract(member, RAW_DATA_DIR)
                    pbar.update(1)
            logger.success(f"Extraction complete! Files extracted to: {RAW_DATA_DIR}")

        if clean:
            zip_file_path.unlink(missing_ok=True)  # Elimina el archivo ZIP si clean es True
            logger.info(f"Removed the ZIP file: {zip_file_path}")
    except zipfile.BadZipFile as e:
        logger.error(f"Error extracting data: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)


@app.command()
def convert_dataset_to_model_format(
    format: DatasetFormat = DatasetFormat.HUGGINGFACE,
    dataset_path: Path = RAW_DATA_EXTRACTION_DIR,
    output_dir: Path = DATASET_CONFIG[DatasetFormat.HUGGINGFACE]["interim_folderpath"],
    clean: bool = False,
):
    match format:
        case DatasetFormat.YOLO:
            if not output_dir:
                logger.debug("No se especificó un directorio de salida, se usará el predeterminado para YOLO")
                output_dir = DATASET_CONFIG[DatasetFormat.YOLO]["interim_folderpath"]
        case DatasetFormat.HUGGINGFACE:
            pass  # Formato por defecto
        case _:
            logger.error(f"Formato {format} no implementado aún")
            raise typer.Exit(1)

    if not dataset_path.exists():
        logger.error(f"Dataset path no existe: {dataset_path}")
        raise typer.Exit(1)

    if not output_dir.exists():
        logger.debug(f"Directorio de salida no existe, se creará: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Convirtiendo dataset a formato {format.value.upper()}")
    logger.info(f"Entrada: {dataset_path}")
    logger.info(f"Salida: {output_dir}")

    # Obtener configuración específica del formato
    config = DATASET_CONFIG.get(format, {})
    logger.debug(f"Configuración del formato: {config}")

    if format == DatasetFormat.YOLO:
        # _convert_to_yolo(dataset_path, output_dir)
        pass
    elif format == DatasetFormat.HUGGINGFACE:
        pass
        # _convert_to_huggingface(dataset_path, output_dir)
    else:
        logger.error(f"Formato {format} no implementado aún")
        raise typer.Exit(1)
    logger.success(f"Conversión completada a formato {format.value.upper()}")

    if clean:
        logger.info(f"Limpiando directorio de entrada: {dataset_path}")
        try:
            shutil.rmtree(dataset_path, ignore_errors=True)
            logger.info(f"Directorio de entrada limpiado: {dataset_path}")
        except Exception as e:
            logger.warning(f"Error al limpiar el directorio de entrada: {e}")
            raise typer.Exit(1)


def _convert_to_yolo(input_path: Path, output_path: Path):
    """Convierte el dataset al formato YOLO"""
    logger.info("Implementando conversión a YOLO...")
    # TODO: Implementar lógica específica de YOLO
    raise NotImplementedError("Conversión a YOLO pendiente de implementar")


def _convert_to_huggingface(input_path: Path, output_path: Path):
    """Convierte el dataset al formato HuggingFace"""
    logger.info("Implementando conversión a HuggingFace...")
    # TODO: Implementar lógica específica de HuggingFace
    raise NotImplementedError("Conversión a HuggingFace pendiente de implementar")


def split_dataset(
    format: DatasetFormat = typer.Option(
        DatasetFormat.HUGGINGFACE, help=f"Formato del dataset. Opciones: {DatasetFormat.list_formats()}"
    ),
    dataset_path: Path = DATASET_CONFIG[DatasetFormat.HUGGINGFACE]["interim_folderpath"],
    output_dir: Path = DATASET_CONFIG[DatasetFormat.HUGGINGFACE]["processed_folderpath"],
    train_ratio: float = 0.8,
):
    """
    Divide el dataset en conjuntos de entrenamiento y validación.

    Args:
        dataset_path: Ruta al dataset procesado
        output_dir: Directorio de salida para los conjuntos divididos
        train_ratio: Proporción del conjunto de entrenamiento (0.0 - 1.0)
    """
    logger.info(
        f"Dividiendo el dataset en {train_ratio*100:.2f}% entrenamiento y {100*(1-train_ratio):.2f}% validación"
    )
    #TODO: Implementar la lógica de división del dataset
    raise NotImplementedError("División del dataset pendiente de implementar")


def load_huggingface_dataset():
    """
    Carga el dataset procesado y lo prepara para su uso.

    Returns:
        Dataset: El dataset cargado y preparado.
    """
    logger.info("Cargando el dataset procesado...")
    # TODO: Implementar la lógica de carga del dataset
    raise NotImplementedError("Carga del dataset pendiente de implementar")


if __name__ == "__main__":
    app()
