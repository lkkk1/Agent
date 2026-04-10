from hashlib import new
from datasets import load_from_disk
from lkkk.pre_train.src import config
from torch.utils.data import DataLoader


def get_dataloader(train = True):
	path = str(config.PROCESSED_DATA_DIR/('train' if train else 'test'))
	datasets= load_from_disk(path)
	# datasets 设置format , 改写get_item 方法，获取结果为torch张量，来兼容支持torch Dataloader
	datasets.set_format(type='torch')
	data_loader = DataLoader(datasets, batch_size=config.BATCH_SIZE, shuffle=True)
	return data_loader

if __name__ == '__main__':
	data_loader = get_dataloader()
	print(len(data_loader))

	for batch in data_loader:
		# batch data 是 字典形式，tokenizer 编码后，是模型需要的输入格式 input_ids , token_type_ids, attention_mask
		for key, value in batch.items():
			print(key, value.shape)
		break