import json


class MixedPrecisionConfig:
    """Layer-to-dtype assignment for mixed-precision quantization.

    Stores a mapping from layer name to quantization dtype and provides
    utilities to construct the config from importance scores, serialize
    it to JSON, and inspect the distribution of dtype assignments.

    Args:
        layer_dtypes: Dict mapping layer name to dtype string ("fp16", "int8", "int4").
    """

    VALID_DTYPES = frozenset({"fp16", "int8", "int4"})

    def __init__(self, layer_dtypes: dict[str, str]):
        invalid = {v for v in layer_dtypes.values() if v not in self.VALID_DTYPES}
        if invalid:
            raise ValueError(f"Invalid dtype(s): {invalid}. Must be one of {self.VALID_DTYPES}")
        self.layer_dtypes = dict(layer_dtypes)

    @classmethod
    def from_importance(
        cls,
        importance: dict[str, float],
        fp16_threshold: float = 0.8,
        int8_threshold: float = 0.5,
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
        layer_dtypes: dict[str, str] = {}
        for name, score in importance.items():
            if score > fp16_threshold:
                layer_dtypes[name] = "fp16"
            elif score > int8_threshold:
                layer_dtypes[name] = "int8"
            else:
                layer_dtypes[name] = "int4"
        return cls(layer_dtypes)

    @classmethod
    def uniform(cls, layer_names: list[str], dtype: str) -> "MixedPrecisionConfig":
        """Create a uniform-precision config where all layers use the same dtype.

        Args:
            layer_names: Names of the layers to include.
            dtype: The dtype to assign to all layers.

        Returns:
            A new MixedPrecisionConfig instance.
        """
        return cls({name: dtype for name in layer_names})

    def to_json(self, path: str):
        """Serialize the config to a JSON file.

        Args:
            path: Destination file path.
        """
        with open(path, "w") as f:
            json.dump(self.layer_dtypes, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MixedPrecisionConfig":
        """Load a MixedPrecisionConfig from a JSON file.

        Args:
            path: Path to a JSON file containing a layer-to-dtype mapping.

        Returns:
            A new MixedPrecisionConfig instance.
        """
        with open(path) as f:
            return cls(json.load(f))

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
