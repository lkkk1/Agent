import torch
from transformers import AutoModelForSequenceClassification

from lkkk.pre_train_bertcls.src import config
from lkkk.pre_train_bertcls.src.dataloader import get_dataloader
from tqdm import tqdm

from lkkk.pre_train_bertcls.src.predict import predict_batch


def evaluate(data_loader, model, device):
	total_count = 0
	acc_count = 0
	for batch in tqdm(data_loader, desc="Evaluating"):
		labels = batch.pop('labels')
		inputs = {k: v.to(device) for k, v in batch.items()}
		predict_res = predict_batch(inputs, model)

		for predict,target in zip(predict_res,labels):
			if predict==target:
				acc_count+=1
			total_count+=1

	return acc_count/total_count



def run_evaluate():
	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

	data_loader = get_dataloader(train=False)

	model = AutoModelForSequenceClassification.from_pretrained(config.MODELS_DIR).to(device)

	acc = evaluate(data_loader, model, device)
	print(f'acc: {acc}')

if __name__ == '__main__':
	run_evaluate()