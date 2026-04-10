from datasets import load_dataset
from lkkk.pre_train.src import config
from datasets import ClassLabel
from transformers import AutoTokenizer

def filter_data(batch):
	# 传入的batch数据是按列组织的，针对review列做过滤
	# results 是一个boolean 列表， 对应每一条记录是否仍要保留
	results = []
	for data in batch['review']:
		results.append(data is not None and data.strip() != '')
	return results



def process():
	print("begin to process")

	# 数据加载
	# data_files 只能接受str
	data_dict = load_dataset('csv', data_files=str(config.RAW_DATA_DIR/'online_shopping_10_cats.csv'))
	datasets = data_dict['train']
	# 数据过滤
	datasets = datasets.remove_columns(['cat'])
	datasets = datasets.filter(filter_data, batched=True)
	print(datasets[1])

	# 划分数据集
	# 按列分层采样，返回的是dict, train, test分别作为key
	# 分层采样的column 只支持 ClassLabel类型， ClassLabel names属性的次序，对应编码值
	datasets = datasets.cast_column('label', ClassLabel(names=['negative', 'positive']))
	print(datasets.features)
	data_dict = datasets.train_test_split(test_size=0.2, shuffle=True, stratify_by_column='label')

	# 数据编码 -- map操作可以由数据集划分后的dict 直接调用，会分别对其中train 和 test 的数据集做编码，返回模型需要的结果
	tokenizers = AutoTokenizer.from_pretrained('bert-base-chinese')

	def batch_encode(batch):
		inputs = tokenizers(batch['review'], padding='max_length',truncation=True, max_length=config.MAX_SEQ_LEN)
		# 模型输入名称是 labels， 为了后边方便解构，所以这里做转换
		inputs['labels'] = batch['label']
		return inputs

	data_dict = data_dict.map(batch_encode, batched=True, remove_columns=['label','review'])
	print(data_dict)
	data_dict.save_to_disk(str(config.PROCESSED_DATA_DIR))
	print("process done")


if __name__ == '__main__':
	process()
