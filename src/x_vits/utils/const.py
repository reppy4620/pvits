from enum import Enum

LRELU_SLOPE = 0.1

JAPANESE_BERT_MODEL_NAME = "globis-university/deberta-v3-japanese-xsmall"
ENGLISH_BERT_MODEL_NAME = "microsoft/deberta-v3-xsmall"


class LANGUAGE(Enum):
    ENGLISH = "EN"
    JAPANESE = "JA"


class PreprocessType(Enum):
    JSUT = "JSUT"

    @classmethod
    def from_str(self, s):
        if s == "JSUT":
            return PreprocessType.JSUT
        else:
            raise ValueError(f"Unknown preprocess type: {s}")
