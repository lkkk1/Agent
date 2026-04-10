import torch
from nltk.translate.bleu_score import corpus_bleu

import Config
from lkkk.nlp.src.BaseTokenizer import ChineseTokenizer, EnglishTokenizer
from lkkk.nlp.src.MethodInputDataSet import get_dataloader
from lkkk.nlp.src.Predict import predict_batch
from lkkk.nlp.src.TranslationModel import TranslationModel


def evaluate(model, test_dataloader, en_tokenizer, device):
	predictions=[]
	references=[]

	for inputs,targets in test_dataloader:
		# inputs.shape [batch_size, seq_len]
		inputs = inputs.to(device)
		targets = targets.tolist()

		batch_result = predict_batch(model, inputs, en_tokenizer)
		predictions.extend(batch_result)
		references.extend([target[1:target.index(en_tokenizer.eos_index)]] for target in targets)

	# bleu 通过n-gram的准确率，计算预测结果和参考译文中的相似度 -- 专门用于翻译比对准确性的
	# 因为references 可以有多个，所以每个预测对应的reference外层多一层【】包裹
	bleu = corpus_bleu(references, predictions)
	return bleu



def run_evaluate():
	# 1. 设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# 2. 分词器
	zh_tokenizer = ChineseTokenizer.from_vocab(Config.MODELS_DIR / 'zh_vocab.txt')
	en_tokenizer = EnglishTokenizer.from_vocab(Config.MODELS_DIR / 'en_vocab.txt')

	# 3. 加载模型
	model = TranslationModel(en_tokenizer.vocab_size, zh_tokenizer.vocab_size, zh_tokenizer.padding_token_index,
	                         en_tokenizer.padding_token_index).to(device)
	model.load_state_dict(torch.load(Config.MODELS_DIR / 'best_model.pt'))
	print("模型加载成功")

	# 4. 数据集
	test_dataloader = get_dataloader(False)

	# 5. 评估逻辑
	bleu = evaluate(model, test_dataloader, en_tokenizer, device)
	print(f"评估结果-blue: {bleu}")


if __name__ == '__main__':
	run_evaluate()

