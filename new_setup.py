from distutils.core import setup
from setuptools import find_packages
import os


# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

REQUIRED = [
    "mrcfile>=1.3",
    "numpy>=1.19",
    "pandas>=1.3.5",
    "pyqtgraph>=0.12.4",
    # "python>=3.9",
    "scipy>=1.7.3",
    "toml>=0.10.2",
    "psutil>=5.9.0",
    "matplotlib>=3.2.2",
    "scikit-image>=0.19.2",
    "pyqt5==5.15.9",
    "qimage2ndarray==1.9.0",
    "cryosparc-tools"
]
NAME = 'grid_edge_detector'
DESCRIPTION = 'A GUI for carbon and other grid edge detection for electron microscopy images.'
URL = 'https://github.com/Croxa/GridEdgeDetector'
EMAIL = 'p.schoennenbeck@fz-juelich.de'
AUTHOR = 'Philipp Schönnenbeck'
REQUIRES_PYTHON = '>=3.9'
VERSION = '0.1.0'

setup(
    # Name of the package
    name=NAME,

    # Packages to include into the distribution
    packages=find_packages('.'), 

    # Start with a small number and increase it with every change you make
    # https://semver.org
    version=VERSION,

    # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    # For example: MIT
    license='MIT',

    # Short description of your library
    description=DESCRIPTION,

    # Long description of your library
    long_description = long_description,
    long_description_context_type = 'text/markdown',

    # Your name
    author=AUTHOR, 

    # Your email
    author_email=EMAIL,     

    entry_points={"console_scripts":["ged=grid_edge_detector.image_gui:main"]},

    # Either the link to your github or to your website
    url=URL,

    # Link from which the project can be downloaded
    # download_url='',

    # List of keyword arguments
    keywords=[],

    # List of packages to install with this one
    install_requires=REQUIRED,

    # https://pypi.org/classifiers/
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',

    ]
)