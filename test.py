from transformers import AutoModel, AutoTokenizer

model_name = "globis-university/deberta-v3-japanese-xsmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

texts = ["こんにちは，私はAIです．"]

inputs = tokenizer(texts, return_tensors="pt", padding=True)
print(inputs)
print(inputs.attention_mask.shape)
print(inputs.attention_mask.sum(dim=1))
