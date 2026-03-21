"""
compress: activation-aware compression for transformer models.

Profiles which attention heads and MLP neurons are most active on a specific
task distribution, then applies more aggressive compression to dormant layers.
"""

from compress.profiling.activation_recorder import ActivationRecorder, ActivationProfile, LayerStats
from compress.profiling.importance import ImportanceScorer
from compress.compression.quantization import ActivationAwareQuantizer, QuantizationConfig, Int8Quantizer, NF4Quantizer
from compress.compression.pruning import HeadPruner
from compress.compression.mixed_precision import MixedPrecisionConfig

__version__ = "0.1.0"
__all__ = [
    "ActivationRecorder",
    "ActivationProfile",
    "LayerStats",
    "ImportanceScorer",
    "ActivationAwareQuantizer",
    "QuantizationConfig",
    "Int8Quantizer",
    "NF4Quantizer",
    "HeadPruner",
    "MixedPrecisionConfig",
]
