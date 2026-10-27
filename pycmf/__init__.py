"""A CMF implementation using PySCF"""

# Add imports here
import numpy as np
import pyscf 

from .pycmf import *
from .helpers import *
from .cmf import *
from .pt2 import QDPT2Result, qdpt2, pair_pt2, pt2_mbe, product_ci, cluster_centroids, cluster_subsets, mbe_increments
from .mrpt import nevpt2, nevpt2_mbe
from . import grad, pdft


try:
    from ._version import __version__
except ImportError:     # source checkout used without installation; `pip install -e .` writes _version.py
    __version__ = '0+unknown'
