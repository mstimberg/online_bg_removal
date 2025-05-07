# Online background removal and cell tracking tool

This tool is used in our lab to record microscope images of Paramecia, remove the background, and write the files to disk. It can optionally track the positions and features, and create a movie of the files in the end. Note that the tool does not take care of controlling the camera itself. Instead, a proprietary camera software continously writes uncompressed TIFF files for each camera image to disk. This software reads these files from the disk, removes the background, and writes new  compressed TIFF file to disk which are usually orders of magnitude smaller. It can also directly delete the original files to avoid running out of disk space for long recordings.

## Table of contents
- [Online background removal and cell tracking tool](#online-background-removal-and-cell-tracking-tool)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [Qt6](#qt6)
    - [CUDA](#cuda)
    - [Getting the code](#getting-the-code)
    - [Installing the dependencies](#installing-the-dependencies)
      - [pixi](#pixi)
      - [conda/mamba](#condamamba)
      - [uv](#uv)
      - [pdm](#pdm)
      - [pip](#pip)
  - [License](#license)
  - [Authors](#authors)


## Installation
### Qt6
The GUI is based on *Qt6*. If you use a conda environment (either directly or via `pixi`), the necessary libraries will be installed automatically. If you use a normal Python virtual environment (e.g. via `uv`, `pdm` or `pip`), then you'll have to make sure that these libraries are installed on your system. For Debian-based Linux distributions, you can install the
libraries with `sudo apt install libqt6widgets6t64`. For other operating systems, have a look at the [Qt documentation](https://doc.qt.io/qt-6/get-and-install-qt.html).

### CUDA
GPU acceleration is optional, but can signficantly improve the processing speed and make the difference for everything running in real-time or not. As for the *Qt* libraries, a CUDA 12.x toolkit is installed automatically if you use `pixi` or `conda/mamba`. If you are not using such an environment, you'll have to take care of install CUDA 12.x yourself, see the [CUDA toolkit documentation](https://docs.nvidia.com/cuda/) for details.

> [!IMPORTANT]
> GPU acceleration is currently only supported for Windows and Linux.

### Getting the code
First, clone the repository, e.g. with
```
$ git clone https://github.com/mstimberg/online_bg_removal.git
```

Alternatively, you can download and unzip the [ZIP archive from GitHub](https://github.com/mstimberg/online_bg_removal/archive/refs/heads/main.zip).

### Installing the dependencies
Below we list various ways of installing the required dependencies for this project. **You will only need to use one of the options!** Note that we recommend using `pixi` or `conda/mamba`, since these environments also take care of installing the required *Qt* library (see above). If you don't have any preference, we recommend using `pixi`.

#### pixi
First, install `pixi` itself if you haven't already, see https://pixi.sh/latest/.

In the project's directory run
```
$ pixi run gui
```
or (to use GPU acceleration)
```
$ pixi run -e gpu gui
```
The first run will set up a conda environment and install the dependencies, later runs will re-use the existing environment.

#### conda/mamba
First, install `mamba` or `conda` itself if you haven't already, e.g. via the [miniforge](https://github.com/conda-forge/miniforge) distribution.

In the project's directory run
```
$ conda env create -f environment.yml
```
or (to use GPU acceleration)
```
$ conda env create -f environment_gpu.yml
```
Use the command stated at the end of the installation to activate the environment and then run
```
$ python background_remover.pyw
```
to start the GUI.


#### uv
First, install `uv` itself if you haven't already, following the [`uv` documentation](https://docs.astral.sh/uv/getting-started/installation/).

In the project's directory run
```
$ uv run background_remover.pyw
```
or (to use GPU acceleration)
```
$ uv run --group gpu background_remover.pyw
```
The first run will set up a virtual environment and install the dependencies, later runs will re-use the existing environment.

#### pdm
First, install `pdm` itself if you haven't already, following the [`pdm` documentation](https://pdm-project.org/en/latest/#installation).

In the project's directory, install the dependencies with
```
$ pdm install --without gpu
```
or (to use GPU acceleration)
```
$ pdm install --with gpu
```

You can then run the GUI with
```
$ pdm run gui
```

#### pip
We highly recommend to first create a virtual environment for the project (note that this needs an existing Python installation with the `venv` module which is sometimes packages separately). In the project's directory run
```
$ python3 -m venv venv
```
And then activate the environment. On Linux
```
$ source venv/bin/activate
```
On Windows
```
$ venv\Scripts\activate
```
Then, install the requirements with either
```
$ pip install -r requirements.txt
```
or
```
$ pip install -r requirements_gpu.txt
```
depending on whether you want to use GPU acceleration or not.


## License
This work is licensed under the European Union Public Licence  v1.2, see the [LICENSE](./LICENSE) file.

## Authors
The tool was written by Marcel Stimberg and Romain Brette at the [Institute of Intelligent Systems and Robotics](https://www.isir.upmc.fr/?lang=en) in Paris, France.
