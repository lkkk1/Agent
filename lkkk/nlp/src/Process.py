import pandas as pd
import Config
from sklearn.model_selection import train_test_split
from lkkk.nlp.src.BaseTokenizer import EnglishTokenizer, ChineseTokenizer


def process():
	print("开始处理数据")
	# 1. 数据读取
	df = pd.read_csv(Config.RAW_DATA_DIR/"cmn.txt", header=None, sep="\t",
	                 usecols=[0,1], names=["en","zh"],encoding="utf-8").dropna()
	# print(df.head())
	# 2. 划分训练集，测试集
	train_datas, test_datas = train_test_split(df, test_size=0.2, random_state=42)

	# 3. 构建词表【按训练集】

	# df 转为python list ，再遍历构建词表，效率更高，如果直接遍历df，底层涉及额外的pandas开销
	# df 转python list 相比 转ndarray , 后者更适合后续有运算需求，如果是单纯构建，前者效率更高
	# sen = train_datas["en"].to_numpy()
	EnglishTokenizer.build_vocab_list(train_datas["en"].tolist(), Config.MODELS_DIR/"en_vocab.txt")
	ChineseTokenizer.build_vocab_list(train_datas["zh"].tolist(), Config.MODELS_DIR/"zh_vocab.txt")

	# 4. 构建训练集，保存训练集
	en_tokenizer = EnglishTokenizer.from_vocab(Config.MODELS_DIR/"en_vocab.txt")
	zh_tokenizer = ChineseTokenizer.from_vocab(Config.MODELS_DIR/"zh_vocab.txt")

	train_datas["zh"] = train_datas["zh"].apply(lambda x: zh_tokenizer.encode(x))
	train_datas["en"] = train_datas["en"].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

	train_datas.to_json(Config.PROCESSED_DATA_DIR/"train.jsonl", orient="records", lines=True)

	# 5. 构建测试集，保存测试集
	test_datas["zh"] = test_datas["zh"].apply(lambda x: zh_tokenizer.encode(x))
	test_datas["en"] = test_datas["en"].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

	test_datas.to_json(Config.PROCESSED_DATA_DIR / "test.jsonl", orient="records", lines=True)

	print("数据处理完毕")


if __name__ == '__main__':
	process()