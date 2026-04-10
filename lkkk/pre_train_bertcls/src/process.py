from datasets import load_dataset
from transformers import AutoTokenizer
from lkkk.pre_train_bertcls.src import config
from datasets import ClassLabel

class DataProcessor():
	_tokenizer = None
	_model_name = 'bert-base-chinese'

	@classmethod
	def _ensure_tokenizer_loaded(cls):
		"""确保tokenizer已加载"""
		if cls._tokenizer is None:
			print(f"🔧 加载tokenizer: {cls._model_name}")
			cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_name)
			print("✅ Tokenizer加载成功")

	@classmethod
	def _batch_filter(cls, batch):
		results = []
		for data in batch['review']:
			results.append(data is not None and data.strip() != '')
		return results

	@classmethod
	def _batch_encode(cls, batch):
		inputs = cls._tokenizer(batch['review'], padding='max_length', truncation=True, max_length=config.MAX_SEQ_LEN)
		inputs['labels'] = batch['label']
		return inputs

	@classmethod
	def process(cls):
		print("processing data start...")

		# 0. 初始化加载tokenizer
		cls._ensure_tokenizer_loaded()

		# 1. 数据加载
		datasets = load_dataset('csv', data_files=str(config.RAW_DATA_DIR / 'online_shopping_10_cats.csv'))['train']
		datasets = datasets.remove_columns('cat')

		# 2. 数据过滤
		datasets = datasets.filter(cls._batch_filter, batched=True)

		# 3. 数据划分
		datasets = datasets.cast_column('label', ClassLabel(names=['negative','positive']))
		data_dict = datasets.train_test_split(test_size=0.2, shuffle=True, stratify_by_column='label')

		# 3. 数据编码
		data_dict = data_dict.map(cls._batch_encode, batched=True, remove_columns=['label', 'review'])

		# 4. 数据保存
		data_dict.save_to_disk(str(config.PROCESSED_DATA_DIR))

		print("processing data done.")


if __name__ == '__main__':
	DataProcessor.process()