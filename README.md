<!-- Improved compatibility of volver al inicio link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->


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
    <!-- TODO: Completar esto. -->
    <!-- <br />
    <a href="https://github.com/brunomaso1/vision-transformer"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/brunomaso1/vision-transformer">View Demo</a>
    &middot;
    <a href="https://github.com/brunomaso1/vision-transformer/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/brunomaso1/vision-transformer/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a> -->
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Tabla de contenido</summary>
  <ol>
    <li>
      <a href="#about-the-project">Sobre el proyecto</a>
      <ul>
        <li><a href="#scafolding">Estructura</a></li>
        <li><a href="#build-with">Herramientas</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Comienzo</a>
      <ul>
        <li><a href="#prerequisites">Prerequisitos</a></li>
        <li><a href="#installation">Instalación</a></li>
      </ul>
    </li>
    <li><a href="#usage">Uso</a></li>
    <li><a href="#contributing">Contribuir</a></li>
    <li><a href="#license">Licencia</a></li>
    <li><a href="#contact">Contacto</a></li>
    <li><a href="#acknowledgments">Agradecimientos</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## 🚀 Sobre el proyecto
<a id="about-the-project"></a>

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

* [![Python][Python.org]][python-url]
* [![Cookiecutter Data Science][cookiecutter-data-science.drivendata.org]][cookiecutter-data-science-url]

> [!WARNING]  
> TODO de acá en delante.

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/brunomaso1/vision-transformer.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
5. Change git remote url to avoid accidental pushes to base project
   ```sh
   git remote set-url origin brunomaso1/vision-transformer
   git remote -v # confirm the changes
   ```

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/brunomaso1/vision-transformer/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>

### Top contributors:

<a href="https://github.com/brunomaso1/vision-transformer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=brunomaso1/vision-transformer" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

Project Link: [https://github.com/brunomaso1/vision-transformer](https://github.com/brunomaso1/vision-transformer)

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">volver al inicio</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[cookiecutter-data-science-url]: https://cookiecutter-data-science.drivendata.org/
[cookiecutter-data-science.drivendata.org]: https://img.shields.io/badge/cookiecutter-data%20science-328F97?logo=cookiecutter&style=for-the-badge

[python-url]: https://www.python.org/
[python.org]: https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white