"""Qwen3 token counting with a cached tokenizer."""
from functools import lru_cache

from dotenv import load_dotenv
from tokenizers import Tokenizer

MODEL = "Qwen/Qwen3-14B"


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    """Load the Qwen3 tokenizer once (uses HF_TOKEN from .env if set)."""
    load_dotenv()
    return Tokenizer.from_pretrained(MODEL)


def count_tokens(text: str) -> int:
    """Number of Qwen3 tokens in text, without special tokens."""
    return len(get_tokenizer().encode(text, add_special_tokens=False).ids)
