"""
KMORFS - Kinetic Modeling Of Residual Film Stress

A physics-informed machine learning framework for modeling residual stress
evolution in thin film materials during Physical Vapor Deposition (PVD).
"""

from .stress_equation import stress_equation
from .data_utils import RawData_extract
from .alloy_extension import AlloyMaterialDependentExtension
from .model import STFModelTorch

__all__ = [
    "stress_equation",
    "RawData_extract",
    "AlloyMaterialDependentExtension",
    "STFModelTorch"
]

__version__ = "1.0.0"
