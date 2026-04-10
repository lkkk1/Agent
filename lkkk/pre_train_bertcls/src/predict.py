import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

from lkkk.pre_train_bertcls.src import config

def predict_batch(inputs, model):
	model.eval()
	with torch.no_grad():
		outputs = model(**inputs)
		logits = outputs.logits
		return torch.argmax(logits, dim=-1).tolist()


def predict(text, model, tokenizer, device):
	inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=config.MAX_SEQ_LEN)
	inputs = {k: v.to(device) for k,v in inputs.items()}
	results = predict_batch(inputs,model)
	return results[0]

def run_predict():
	# 设备
	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

	# 加载模型
	model = AutoModelForSequenceClassification.from_pretrained(config.MODELS_DIR).to(device)
	print(type(model))

	# tokenizer
	tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

	while True:
		input_text = input("> ")
		if input_text in ["exit",'q', 'quit']:
			break
		if input_text.strip() == "":
			print("请输入文本")
			continue

		predict_res = predict(input_text, model, tokenizer, device)
		print('positive' if predict_res==1 else 'negative')


if __name__ == "__main__":
	run_predict()