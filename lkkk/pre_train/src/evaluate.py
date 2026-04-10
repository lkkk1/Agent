import torch
from lkkk.pre_train.src import config
from lkkk.pre_train.src.classifymodel import classifymodel
from lkkk.pre_train.src.dataloader import get_dataloader
from lkkk.pre_train.src.predict import predict_batch
from tqdm import tqdm


def evaluate(model, test_dataloader, device):
	total_count = 0
	acc_count = 0
	for inputs in tqdm(test_dataloader, desc="Evaluating"):
		targets = inputs.pop('labels')
		inputs = {k: v.to(device) for k,v in inputs.items()}
		predict_res_batch = predict_batch(model, inputs)

		for predict, target in zip(predict_res_batch, targets):
			predict = 1 if predict > 0.5 else 0
			if predict == target:
				acc_count += 1
			total_count += 1
	return acc_count / total_count


def run_evaluate():
	# 设备
	device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

	# 模型
	model = classifymodel().to(device)
	model.load_state_dict(torch.load(config.MODELS_DIR/'best.pt'))
	print("模型加载成功")

	# 数据集
	test_dataloader = get_dataloader(train=False)

	# 评估逻辑
	acc = evaluate(model, test_dataloader, device)
	print(f'acc:{acc}')



if __name__ == '__main__':
	run_evaluate()