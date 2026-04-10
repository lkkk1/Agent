import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm
from lkkk.nlp.src.MethodInputDataSet import get_dataloader
from lkkk.nlp.src.TranslationModel import TranslationModel
from lkkk.nlp.src.BaseTokenizer import EnglishTokenizer, ChineseTokenizer
from lkkk.nlp.src import Config


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
	model.train()
	total_loss = 0
	for data, target in tqdm(dataloader):
		# inputs.shape [batch_size, seq_len]
		# targets.shape [batch_size, tgt_len]
		encoder_inputs = data.to(device)
		target = target.to(device)

		# decoder_inputs.shape [batch_size, seq_len], 去掉eos
		decoder_inputs = target[:,:-1]
		# decoder_targets.shape [batch_size, seq_len], 去掉sos
		decoder_targets = target[:,1:]

		# 前向传播
		src_pad_mask = (encoder_inputs == model.zh_embed.padding_idx)
		tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1]).to(device)

		# decoder_outpus.shape [batch_size, seq_len, vocab_size]
		decoder_outputs = model(encoder_inputs, decoder_inputs, src_pad_mask, tgt_mask)

		# 计算loss 匹配shape , NC 和 N
		loss = loss_fn(decoder_outputs.reshape(-1, decoder_outputs.shape[-1]), decoder_targets.reshape(-1))

		# 反向传播
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		total_loss += loss.item()

	return total_loss / len(dataloader)


def train():
	# 1. 确定设备
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# 2. dataloader
	data_loader = get_dataloader()

	# 3. tokenizer
	en_tokenizer = EnglishTokenizer.from_vocab(Config.MODELS_DIR/'en_vocab.txt')
	zh_tokenizer = ChineseTokenizer.from_vocab(Config.MODELS_DIR/'zh_vocab.txt')

	# 4. 模型
	model = TranslationModel(en_tokenizer.vocab_size, zh_tokenizer.vocab_size, zh_tokenizer.padding_token_index, en_tokenizer.padding_token_index).to(device)

	# 5. loss
	# loss计算的时候，如果已经是eos后边的pad, 对loss计算不影响
	loss_fn = nn.CrossEntropyLoss(ignore_index=en_tokenizer.padding_token_index)

	# 6. 优化器
	optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARN_RATE)

	# 7. Tensorboard writer
	writer = SummaryWriter(Config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S"))


	# 6. 开始训练
	best_loss = float('inf')
	for epoch in range(Config.EPOCHS):
		print("="*10, f"Epoch {epoch+1}/{Config.EPOCHS}", "="*10)
		loss = train_one_epoch(model, data_loader, optimizer, loss_fn, device)
		print(f'Train loss: {loss:.4f}')

		writer.add_scalar('loss', loss, epoch)

		# 保存模型
		if loss < best_loss:
			best_loss = loss
			torch.save(model.state_dict(), Config.MODELS_DIR/"best_model.pt")
			print("保存模型")
		writer.close()

if __name__ == '__main__':
	train()