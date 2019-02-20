import os
from pathlib import Path

from setuptools import setup

THIS_DIR = os.path.abspath(os.path.dirname(__file__))

setup(
    name = 'tfrt',
    version = "0.1.0",
    long_description = Path('README.md').read_text(),
    long_description_content_type = "text/markdown",
    url = 'https://github.com/ecpoppenheimer/TensorFlowRayTrace',
    classifiers = [
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    packages = [
        'tfrt',
    ],
    install_requires = Path('requirements.txt').read_text().splitlines(),
)
