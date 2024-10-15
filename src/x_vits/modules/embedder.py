import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from x_vits.utils.const import ENGLISH_BERT_MODEL_NAME, JAPANESE_BERT_MODEL_NAME, LANGUAGE


class ContextEmbedder(nn.Module):
    def __init__(self, language=LANGUAGE.JAPANESE):
        super().__init__()
        if language == LANGUAGE.JAPANESE:
            model_name = JAPANESE_BERT_MODEL_NAME
        elif language == LANGUAGE.ENGLISH:
            model_name = ENGLISH_BERT_MODEL_NAME
        else:
            raise ValueError(f"Invalid language: {language}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.register_buffer("dummy", torch.tensor(0))

    @property
    def device(self):
        return self.dummy.device

    def forward(self, raw_texts):
        inputs = self.tokenizer(raw_texts, return_tensors="pt", padding=True).to(self.device)
        context_lengths = inputs.attention_mask.sum(dim=1)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, context_lengths
