from datasets import load_from_disk
from lkkk.pre_train_bertcls.src import config
from torch.utils.data import DataLoader

def get_dataloader(train=True):
	path = config.PROCESSED_DATA_DIR/('train' if train else 'test')
	datasets = load_from_disk(str(path))
	datasets.set_format(type='torch')
	data_loader = DataLoader(datasets, batch_size=config.BATCH_SIZE, shuffle=True)
	return data_loader

if __name__ == '__main__':
	data_loader = get_dataloader()
	print(len(data_loader))
	for batch in data_loader:
		for key, value in batch.items():
			print(key, value.shape)
		break