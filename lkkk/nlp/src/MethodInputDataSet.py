from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from lkkk.nlp.src import Config
import torch


class MethodInputDataSet(Dataset):
	def __init__(self, data_path):
		# dataframe .to_dict 方法，orient=records, 最常用，保留序列整体k-v结构
		self.data = pd.read_json(data_path, encoding='utf-8', lines=True, orient='records').to_dict(orient='records')


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		input_tensor = torch.tensor(self.data[idx]['zh'], dtype=torch.long)
		target_tensor = torch.tensor(self.data[idx]['en'], dtype=torch.long)
		return input_tensor, target_tensor

def collator(batch):
	# batch: 一个列表，每个元素是Dataset.__getitem__返回的一个样本
	input_tensors = [item[0] for item in batch]
	target_tensors = [item[1] for item in batch]

	input_tensors=pad_sequence(input_tensors, batch_first=True, padding_value=0)
	target_tensors=pad_sequence(target_tensors, batch_first=True, padding_value=0)
	return input_tensors, target_tensors


def get_dataloader(train=True):
	path = Config.PROCESSED_DATA_DIR/('train.jsonl' if train else 'test.jsonl')
	dataset = MethodInputDataSet(path)

	# 对于变长的序列，如果不做collate_fn处理，在同一列，长度不同，张量无法堆叠，需要collate_fn方法中做pad处理，填充到对应列最大长度
	return DataLoader(dataset, shuffle=True, batch_size=Config.BATCH_SIZE, collate_fn=collator)

if __name__ == "__main__":
	dataset = MethodInputDataSet(Config.PROCESSED_DATA_DIR/"train.jsonl")
	print(type(dataset))
	print(dataset.__getitem__(1))