import json
import math

from .quantization import QuantizationConfig


class MixedPrecisionConfig:
    """Layer-to-dtype assignment for mixed-precision quantization.

    Stores a mapping from layer name to quantization dtype and provides
    utilities to construct the config from importance scores, serialize
    it to JSON, and inspect the distribution of dtype assignments.

    Args:
        layer_dtypes: Dict mapping layer name to dtype string ("fp16", "int8", "int4").
    """

    VALID_DTYPES = frozenset({"fp16", "int8", "int4"})

    def __init__(self, layer_dtypes: dict[str, str], *, protect_output_head: bool = True,
                 output_head_names: tuple[str, ...] = ("lm_head",)):
        invalid = {v for v in layer_dtypes.values() if v not in self.VALID_DTYPES}
        if invalid:
            raise ValueError(f"Invalid dtype(s): {invalid}. Must be one of {self.VALID_DTYPES}")
        config = QuantizationConfig(layer_dtypes, protect_output_head=protect_output_head,
                                    output_head_names=output_head_names)
        self.layer_dtypes = {name: config.dtype_for(name) for name in layer_dtypes}
        self.protect_output_head = protect_output_head
        self.output_head_names = tuple(output_head_names)

    def to_quantization_config(self, default_dtype="int8") -> QuantizationConfig:
        return QuantizationConfig(self.layer_dtypes, default_dtype, self.protect_output_head,
                                  self.output_head_names)

    @classmethod
    def from_importance(
        cls,
        importance: dict[str, float],
        fp16_threshold: float = 0.8,
        int8_threshold: float = 0.5,
        *,
        protect_output_head: bool = True,
        output_head_names: tuple[str, ...] = ("lm_head",),
    ) -> "MixedPrecisionConfig":
        """Build a MixedPrecisionConfig from normalized importance scores.

        Args:
            importance: Dict mapping layer name to importance score in [0, 1].
            fp16_threshold: Layers with score above this threshold get FP16.
            int8_threshold: Layers with score above this (but ≤ fp16_threshold) get INT8.
                            Layers at or below this threshold get INT4.

        Returns:
            A new MixedPrecisionConfig instance.
        """
        if not 0 <= int8_threshold <= fp16_threshold <= 1:
            raise ValueError("Thresholds must satisfy 0 <= int8 <= fp16 <= 1")
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in importance.values()):
            raise ValueError("Importance scores must be finite and in [0, 1]")
        layer_dtypes: dict[str, str] = {}
        for name, score in importance.items():
            if score > fp16_threshold:
                layer_dtypes[name] = "fp16"
            elif score > int8_threshold:
                layer_dtypes[name] = "int8"
            else:
                layer_dtypes[name] = "int4"
        return cls(layer_dtypes, protect_output_head=protect_output_head,
                   output_head_names=output_head_names)

    @classmethod
    def uniform(cls, layer_names: list[str], dtype: str, *, protect_output_head: bool = True,
                output_head_names: tuple[str, ...] = ("lm_head",)) -> "MixedPrecisionConfig":
        """Create a uniform body plan with the declared output-head floor.

        Set protect_output_head=False for an explicitly all-NF4 experiment.

        Args:
            layer_names: Names of the layers to include.
            dtype: The dtype to assign to all layers.

        Returns:
            A new MixedPrecisionConfig instance.
        """
        return cls({name: dtype for name in layer_names}, protect_output_head=protect_output_head,
                   output_head_names=output_head_names)

    def to_json(self, path: str):
        """Serialize the config to a JSON file.

        Args:
            path: Destination file path.
        """
        with open(path, "w") as f:
            json.dump({"format_version": 2, "policy": self.to_quantization_config().to_dict()},
                      f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MixedPrecisionConfig":
        """Load a MixedPrecisionConfig from a JSON file.

        Args:
            path: Path to a JSON file containing a layer-to-dtype mapping.

        Returns:
            A new MixedPrecisionConfig instance.
        """
        with open(path) as f:
            value = json.load(f)
        if isinstance(value, dict) and "format_version" in value:
            if value["format_version"] != 2 or set(value) != {"format_version", "policy"}:
                raise ValueError("Unsupported precision plan format")
            config = QuantizationConfig.from_dict(value["policy"])
            return cls(config.layer_dtypes, protect_output_head=config.protect_output_head,
                       output_head_names=config.output_head_names)
        # Historical mapping-only plans retain their recorded assignments.
        return cls(value, protect_output_head=False)

    def summary(self) -> dict[str, int]:
        """Return the count of layers assigned to each dtype.

        Returns:
            Dict mapping dtype string to number of layers.
        """
        counts: dict[str, int] = {"fp16": 0, "int8": 0, "int4": 0}
        for dtype in self.layer_dtypes.values():
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts

    def get(self, layer_name: str, default: str = "int8") -> str:
        """Return the dtype for a given layer, or default if not found."""
        return self.layer_dtypes.get(layer_name, default)

    def __repr__(self) -> str:
        s = self.summary()
        return f"MixedPrecisionConfig(fp16={s['fp16']}, int8={s['int8']}, int4={s['int4']})"
