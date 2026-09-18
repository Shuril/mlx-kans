"""
metal-KANs: Pure Metal Shading Language (MSL) Kolmogorov-Arnold Networks for Apple Silicon.
High-throughput, zero-allocation GPU compute shaders with Python/NumPy interface.
"""

from .metal_kan import (
    MetalChebyKAN,
    MetalFastKAN,
    MetalReLUKAN,
    MetalKAN,
)

__all__ = [
    "MetalChebyKAN",
    "MetalFastKAN",
    "MetalReLUKAN",
    "MetalKAN",
]
