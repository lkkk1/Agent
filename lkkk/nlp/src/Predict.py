import torch

from lkkk.nlp.src.BaseTokenizer import ChineseTokenizer, EnglishTokenizer
import Config
from lkkk.nlp.src.TranslationModel import TranslationModel

def predict_batch(model, input_tensor, en_tokenizer):
	"""
	:param model: 模型
	:param input_tensor:  [batch_size, seq_length]
	:param en_tokenizer:
	:return: 预测结果
	"""
	model.eval()
	with (torch.no_grad()):
		# 编码
		src_pad_mask = (input_tensor == model.zh_embed.padding_idx)
		memory = model.encode(input_tensor, src_pad_mask)
		# memory shape [batch_size, src_seq_len, dim_model]

		# 解码
		batch_size = input_tensor.shape[0]
		device = input_tensor.device

		# 构建解码器，第一个token的输入
		decoder_input = torch.full([batch_size, 1], en_tokenizer.sos_token_index, device=device)
		# shape [batch_size, 1]

		# 预测结果缓存
		generated = []

		# 记录每个样本是否已经生成结束符
		is_finished = torch.full([batch_size], False, device=device)

		# 自回归生成 -- 生成过程是顺序的，每次要重新更新输入，带上已经生成的部分 -- tensorflow 和 rnn的区别
		for i in range(Config.MAX_SEQ_LEN):
			tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1])
			decoder_output = model.decode(decoder_input, memory, tgt_mask, src_pad_mask)
			# decoder_output shape [batch_size, tgt_seq_len, en_vocab_size]

			# 保存预测结果
			next_token_indexes = torch.argmax(decoder_output[:,-1,:], dim=-1, keepdim=True)
			# next_token_indexes shape [batch_size, 1]
			generated.append(next_token_indexes)

			# 更新输入
			decoder_input = torch.cat((decoder_input, next_token_indexes), dim=-1)

			# 判断是否应该结束
			# 由于每个输入的长短不同，对于短的句子，如果前边已经有eos,已经是true了，后边长句子判断的时候，短句子再判断，所以要加或
			is_finished |= (next_token_indexes.squeeze(1) == en_tokenizer.eos_index)
			if is_finished.all():
				break

		# 处理预测结果
		# generated shape [tensor([batch_size, 1])]
		generated_tensor = torch.cat(generated, dim=1)
		# generated_tensor shape [batch_size, seq_len]

		generated_list = generated_tensor.tolist()
		# 去掉eos后边的token id
		for index,sentence in enumerate(generated_list):
			if en_tokenizer.eos_index in sentence:
				eos_pos = sentence.index(en_tokenizer.eos_index)
				generated_list[index] = sentence[:eos_pos]
		return generated_list

def predict(src, model, en_tokenizer, zh_tokenizer, device):
	# 处理输入
	indexes = zh_tokenizer.encode(src)
	input_tensor = torch.tensor(indexes, dtype=torch.long).unsqueeze(0).to(device)
	# 预测
	batch_result = predict_batch(model, input_tensor, en_tokenizer)
	return en_tokenizer.decode(batch_result[0])

def run_predict():
	# 1. 设备
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# 2. 分词器
	zh_tokenizer = ChineseTokenizer.from_vocab(Config.MODELS_DIR/'zh_vocab.txt')
	en_tokenizer = EnglishTokenizer.from_vocab(Config.MODELS_DIR/'en_vocab.txt')

	# 3. 加载模型
	model = TranslationModel(en_tokenizer.vocab_size, zh_tokenizer.vocab_size, zh_tokenizer.padding_token_index, en_tokenizer.padding_token_index).to(device)
	model.load_state_dict(torch.load(Config.MODELS_DIR/'best_model.pt'))
	print("模型加载成功")

	# 4. 输入预测
	while True:
		user_input = input("请输入中文：")
		if user_input in ["exit",'q','quit']:
			print("exit")
			break
		if user_input.strip() == "":
			print("请输入内容")
			continue
		result = predict(user_input, model,en_tokenizer,zh_tokenizer, device)
		print("英文：",result)

if __name__ == '__main__':
	run_predict()