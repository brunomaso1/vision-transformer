<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/brunomaso1/vision-transformer">
    <img src="resources/logo.png" alt="Logo" width="200" height="80">
  </a>

<h3 align="center">Vision Transformer</h3>
  <p align="center">
    <div>
      <table align="center">
        <tr>
          <th>Subtitulo</th>
          <td>Trabajo final VPC3</td>
        </tr>
        <tr>
          <th>Descripción</th>
          <td>Casificación de imágenes satelitales utilizando <em>vision transformers</em></td>
        </tr>
        <tr>
          <th>Integrantes</th>
          <td>- Bruno Masoller (brunomaso1@gmail.com)<br/>- Juan Cruz Ferreyra (ferreyra.juancruz95@gmail.com)<br/>- Mauro Aguirregaray (maguirregaray13cb@gmail.com)</td>
          <td>
        </tr>
      </table>
    </div>
    <br />
    <a href="https://github.com/brunomaso1/vision-transformer"><strong>Explora la documentación »</strong></a>
    <br />
    <br />
    <a href="https://github.com/brunomaso1/vision-transformer">Ver una demo</a>
    &middot;
    <a href="https://github.com/brunomaso1/vision-transformer/issues/new?labels=bug&template=bug-report---.md">Reportar un problema</a>
    &middot;
    <a href="https://github.com/brunomaso1/vision-transformer/issues/new?labels=enhancement&template=feature-request---.md">Solicitar una característica</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Tabla de contenido</summary>
  <ol>
    <li>
      <a href="#about-the-project">🚀 Sobre el proyecto</a>
      <ul>
        <li><a href="#scafolding">📝 Estructura</a></li>
        <li><a href="#build-with">🛠️ Herramientas</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">🏁 Comienza aquí</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Instalación</a></li>
      </ul>
    </li>
    <li><a href="#usage">⚡ Uso</a></li>
    <li><a href="#roadmap">🛣️ Roadmap</a></li>
    <li><a href="#contributing">👨‍👩‍👧‍👦 Contribuir</a></li>
    <li><a href="#contributors">Mejores contribuyentes</a></li>
    <li><a href="#license">⚠️ Licencia</a></li>
    <li><a href="#contact">📞 Contacto</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## 🚀 Sobre el proyecto
<a id="about-the-project"></a>

<!-- TODO:
* Poner una imagen de inferencia de un modelo. -->

Este proyecto tiene como objetivo clasificar imágenes satelitales utilizando <em>vision transformers</em>. Se posiciona en el marco de un trabajo final para la materia "Visión por Computadora 3" del posgrado en Inteligencia Artificial de la Facultad de Ingeniería de la Universidad de Buenos Aires.

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

### 📝 Estructura
<a id="scafolding"></a>

El proyecto utiliza la plantilla [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/) para organizar el código y los datos de manera eficiente. Esta estructura ayuda a mantener el proyecto limpio y escalable, facilitando la colaboración y el mantenimiento.

La estructura del proyecto es la siguiente:

```
├── LICENSE            <- Licencia del proyecto.
├── Makefile           <- Makefile con comandos útiles como `make data` o `make train`.
├── README.md          <- El README principal para desarrolladores que usan este proyecto.
├── CHANGELOG.md       <- Registro de cambios del proyecto.
├── data
│   ├── external       <- Datos de fuentes de terceros.
│   ├── interim        <- Datos intermedios que han sido transformados.
│   ├── processed      <- Conjuntos de datos finales y canónicos para el modelado.
│   └── raw            <- Volcado de datos original e inmutable.
│
├── docs               <- Proyecto mkdocs por defecto; ver www.mkdocs.org para más detalles.
│
├── models             <- Modelos entrenados y serializados, predicciones o resúmenes de modelos.
│
├── notebooks          <- Notebooks de Jupyter. Convención de nombres: un número (para ordenar),
│                         las iniciales del creador y una breve descripción separada por `-`, por ejemplo
│                         `1.0-jqp-exploracion-inicial-datos`.
│
├── pyproject.toml     <- Archivo de configuración del proyecto con metadatos del paquete para 
│                         vision_transformer y configuración para herramientas como black.
│
├── references         <- Diccionarios de datos, manuales y otros materiales explicativos.
│
├── reports            <- Análisis generados en HTML, PDF, LaTeX, etc.
│   └── figures        <- Gráficos y figuras generados para usar en los reportes.
│
|── resources          <- Recursos adicionales como imágenes, archivos de audio, etc.      
|
├── requirements.txt   <- Archivo de requisitos para reproducir el entorno de análisis, por ejemplo,
│                         generado con `pip freeze > requirements.txt`.
│
├── setup.cfg          <- Archivo de configuración para flake8.
│
└── vision_transformer   <- Código fuente utilizado en este proyecto.
    │
    ├── __init__.py             <- Convierte vision_transformer en un módulo de Python.
    │
    ├── config.py               <- Almacena variables y configuraciones útiles.
    │
    ├── dataset.py              <- Scripts para descargar o generar datos.
    │
    ├── features.py             <- Código para crear características para el modelado.
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Código para ejecutar inferencia con modelos entrenados.          
    │   └── train.py            <- Código para entrenar modelos.
    │
    └── plots.py                <- Código para crear visualizaciones.
```

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

### 🛠️ Herramientas
<a id="build-with"></a>

Este proyecto está construido utilizando una variedad de herramientas entre las cuales se incluyen:

* [![Python][python.org]][python-url]
* [![Cookiecutter Data Science][cookiecutter-data-science.drivendata.org]][cookiecutter-data-science-url]
* [![Poetry][python-poetry.org]][python-poetry-url]

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

<!-- GETTING STARTED -->
## 🏁 Comienza aquí
<a id="getting-started"></a>

Este es un ejemplo de cómo puedes dar instrucciones para configurar tu proyecto localmente.
Para obtener una copia local y ejecutarla, sigue estos sencillos pasos de ejemplo.

### Prerequisites
<a id="prerequisites"></a>

* **Poetry**: Asegúrate de tener instalado [Poetry](https://python-poetry.org/docs/#installation) para gestionar las dependencias del proyecto.
* **Python**: Este proyecto requiere Python 3.13 o superior. Puedes descargarlo desde [python.org](https://www.python.org/downloads/).

### Instalación
<a id="installation"></a>

1. Fork the Project
   - Ve a la página del proyecto en GitHub y haz clic en el botón "Fork" para crear una copia del repositorio en tu cuenta.
   - Alternativamente, puedes clonar el repositorio directamente si no deseas hacer un fork mediante el siguiente comando:
     - `bash`:
        ```sh
        git clone https://github.com/brunomaso1/vision-transformer.git
        ```
2. Con Poetry, instala las dependencias del proyecto:
    - `bash`:
   ```sh
   poetry install
   ```
3. Activa el entorno virtual via:
     - `powershell`:
   ```powerShell
   Invoque-Expresion(poetry env activate)
   ```
     - `bash`:
   ```bash
   eval $(poetry env activate)
   ```

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

<!-- USAGE EXAMPLES -->
## ⚡ Uso
<a id="usage"></a>

> [!WARNING]  
> 🚧 Sección en construcción... 🚧

<!-- TODO:
* Explicar cómo ejecutar el proyecto, por ejemplo, cómo entrenar un modelo o hacer inferencias.
* Incluir ejemplos de comandos que se pueden ejecutar en la terminal.
* Incluir ejemplos de cómo utilizar las funciones principales del proyecto.

_Para más ejemplos, consulta la [Documentación](https://example.com)_
 -->

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

<!-- ROADMAP -->
## 🛣️ Roadmap
<a id="roadmap"></a>

- [ ] Mejorar la documentación del proyecto.
- [ ] Crear el pipeline de entrenamiento y evaluación de modelos.
- [ ] Implementar un modelo de <em>vision transformer</em> para clasificación de imágenes.

Consulta la [lista de issues abiertas](https://github.com/brunomaso1/vision-transformer/issues) para ver todas las características propuestas (y problemas conocidos).

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

<!-- CONTRIBUTING -->
## 👨‍👩‍👧‍👦 Contribuir
<a id="contributing"></a>

Las contribuciones son lo que hace que la comunidad de código abierto sea un lugar increíble para aprender, inspirar y crear. **¡Cualquier contribución que realices será muy apreciada!**

Si tienes una sugerencia para mejorar este proyecto, por favor haz un fork del repositorio y crea un pull request. También puedes abrir un issue con la etiqueta "enhancement".  
¡No olvides darle una estrella al proyecto! ¡Gracias nuevamente!

1. Haz un fork del proyecto
2. Crea una rama para tu funcionalidad (`git checkout -b feature/FuncionalidadAsombrosa`)
3. Realiza tus cambios y haz commit (`git commit -m 'Agrega FuncionalidadAsombrosa'`)
4. Haz push a la rama (`git push origin feature/FuncionalidadAsombrosa`)
5. Abre un Pull Request

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

### Mejores contribuyentes
<a id="contributors"></a>

<a href="https://github.com/brunomaso1/vision-transformer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=brunomaso1/vision-transformer" alt="contrib.rocks image" />
</a>

<!-- LICENSE -->
## ⚠️ Licencia
<a id="license"></a>

Distribuido bajo la licencia GNU. Consulta `LICENSE` para más información.

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

<!-- CONTACT -->
## 📞 Contacto
<a id="contact"></a>

- Bruno Masoller (brunomaso1@gmail.com)
- Juan Cruz Ferreyra (ferreyra.juancruz95@gmail.com)
- Mauro Aguirregaray (maguirregaray13cb@gmail.com)

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

<!-- ACKNOWLEDGMENTS -->
<!-- TODO:
* Agradecimientos.
## 🫂 Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[cookiecutter-data-science-url]: https://cookiecutter-data-science.drivendata.org/
[cookiecutter-data-science.drivendata.org]: https://img.shields.io/badge/cookiecutter-data%20science-328F97?logo=cookiecutter&style=for-the-badge

[python-url]: https://www.python.org/
[python.org]: https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-poetry-url]: https://python-poetry.org/
[python-poetry.org]: https://img.shields.io/badge/poetry-000000?style=for-the-badge&logo=poetry&logoColor=white