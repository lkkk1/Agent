import torch
from transformers import AutoTokenizer
from lkkk.pre_train.src import config
from lkkk.pre_train.src.classifymodel import classifymodel

def predict_batch(model, inputs):
	model.eval()
	with torch.no_grad():
		outputs = model(**inputs)

	# 针对二分类任务，输出结果通过sigmoid 映射到 【0，1】的范围内，可以通过与0.5比值判断正负
	batch_result = torch.sigmoid(outputs)
	return batch_result.tolist()

def predict(text, model, tokenizer, device):
	# 处理输入
	inputs = tokenizer(text, padding='max_length', truncate=True, return_tensors='pt', max_length=config.MAX_SEQ_LEN)
	inputs = {k: v.to(device) for k,v in inputs.items()}

	batch_result = predict_batch(model, inputs)
	return batch_result[0]



def run_predict():
	# 设备
	device = torch.device("mps" if torch.cuda.is_available() else "cpu")

	# tokenizer
	tokenizers = AutoTokenizer.from_pretrained("bert-base-chinese")

	# 加载模型
	model = classifymodel().to(device)
	model.load_state_dict(torch.load(config.MODELS_DIR/'best.pt'))
	print("模型加载成功")

	while True:
		user_input = input("> ")
		if user_input in ["exit", "quit", 'q']:
			print("exit")
			break
		if user_input.strip() == "":
			print("请输入内容")
			continue
		predict_res = predict(user_input, model, tokenizers, device)
		print('positive' if predict_res > 0.5 else 'negative')


if __name__ == "__main__":
	run_predict()