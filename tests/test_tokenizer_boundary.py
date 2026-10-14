"""Exercise the dependency boundary used when quantize saves a Hub tokenizer."""
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


def test_tokenizer_save_rejects_chat_template_path_traversal(tmp_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.chat_template = {"default": "safe", "../../outside": "untrusted template"}
    with pytest.raises(ValueError, match="template"):
        tokenizer.save_pretrained(tmp_path / "saved")
    assert not (tmp_path / "outside.jinja").exists()
