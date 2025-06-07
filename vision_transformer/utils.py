# Enum para los formatos de dataset
from enum import Enum


class DatasetFormat(Enum):
    """
    Enum para representar los formatos de dataset soportados.
    Atributos:
        YOLO: Representa el formato de dataset YOLO.
        HUGGINGFACE: Representa el formato de dataset HuggingFace.
    Métodos:
        __str__(): Retorna el valor del formato como string.
        list_formats(): Retorna una lista de todos los formatos disponibles.
        from_string(format_str: str): Crea una instancia del enum a partir de un string (case-insensitive).
    Ejemplo de uso:
        >>> formato = DatasetFormat.from_string("yolo")
        >>> print(formato)  # Salida: yolo
    """

    YOLO = "yolo"
    HUGGINGFACE = "huggingface"

    def __str__(self):
        return self.value

    @classmethod
    def list_formats(cls):
        """Retorna una lista de todos los formatos disponibles"""
        return [fmt.value for fmt in cls]

    @classmethod
    def from_string(cls, format_str: str):
        """Crea un enum desde un string, case-insensitive"""
        format_str = format_str.lower()
        for fmt in cls:
            if fmt.value.lower() == format_str:
                return fmt
        raise ValueError(f"Formato '{format_str}' no válido. Formatos disponibles: {cls.list_formats()}")
