from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import List


class Tokenizer:
    """
    Tokenizer class
    """
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def n_words(self) -> int:
        return self._model.vocab_size()

    @property
    def bos_id(self) -> int:
        return self._model.bos_id()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str, bos: bool = True) -> List[int]:
        """
        Encode a text string into a list of token IDs.

        Args:
            s (str): The input text to be encoded.
            bos (bool, optional): Whether to prepend a beginning-of-sequence token. Defaults to True.

        Returns:
            List[int]: A list of token IDs representing the encoded text.
        """
        assert isinstance(s, str)
        t = self._model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decode a list of token IDs into a text string.

        Args:
            t (List[int]): A list of token IDs to be decoded.

        Returns:
            str: The decoded text string.
        """
        return self._model.decode(t)
