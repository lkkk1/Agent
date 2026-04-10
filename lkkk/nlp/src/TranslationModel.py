import math
from torch import nn
import torch
from lkkk.nlp.src import Config


class PositionEncoding(nn.Module):
	def __init__(self, dim_model, max_len):
		super(PositionEncoding, self).__init__()
		self.position = torch.zeros(max_len,dim_model)

		for pos in range(max_len):
			for _2i in range(0,dim_model,2):
				self.position[pos,_2i] = math.sin(pos/(10000 ** (_2i/dim_model)))
				self.position[pos,_2i+1] = math.cos(pos/(10000 ** (_2i/dim_model)))
		self.register_buffer('pe',self.position)

	def forward(self, x):
		# x.shape [batch_size, seq_len, dim_model]
		seq_len = x.shape[1]
		part_pe = self.pe[:seq_len, :]
		# part_pe.shape [seq_len, dim_model]
		part_pe = part_pe.unsqueeze(0)

		# 两个张量相加时，PyTorch 会从最右边的维度开始逐维比较：
		# 最右边维度：dim_modelvs dim_model→ 相等 ✓
		# 中间维度：seq_lenvs seq_len→ 相等 ✓
		# 最左边维度：batch_sizevs 缺失​ → 不匹配 ❌
		# 当某个维度缺失时，PyTorch 会尝试自动扩展，但只能从维度大小为1或缺失的维度开始扩展。
		# 这里x做位置编码，其实输入是1维，所以part_pe可以不做增加维度处理，但安全起见保留
		return x + part_pe


class TranslationModel(nn.Module):
	def __init__(self, en_vocab_size, zh_vocab_size, zh_padding_index, en_padding_index):
		super(TranslationModel, self).__init__()
		self.en_embed = nn.Embedding(num_embeddings=en_vocab_size,
		                             embedding_dim=Config.DIM_MODEL,
		                             padding_idx=en_padding_index)
		self.zh_embed = nn.Embedding(num_embeddings=zh_vocab_size,
		                             embedding_dim=Config.DIM_MODEL,
		                             padding_idx=zh_padding_index)

		# 位置编码层
		self.position_encoding = PositionEncoding(dim_model=Config.DIM_MODEL, max_len=Config.MAX_SEQ_LEN)

		self.transformer = nn.Transformer(d_model=Config.DIM_MODEL,
		                                  nhead=Config.NUM_HEADS,
		                                  num_decoder_layers=Config.NUM_DECODER_LAYERS,
		                                  num_encoder_layers=Config.NUM_ENCODER_LAYERS,
		                                  batch_first=True)

		self.linear = nn.Linear(in_features=Config.DIM_MODEL, out_features=en_vocab_size)


	def forward(self, src, tgt, src_pad_mask, tgt_mask):
		memory = self.encode(src, src_pad_mask)
		output = self.decode(tgt, memory, tgt_mask, src_pad_mask)
		return output

	def encode(self, src, src_pad_mask):
		# src.shape [batch_size, seq_len]
		# src_pad_mask.shape [batch_size, seq_len] -- true,false 指定哪些位置是pad, 在编码计算自注意力相关分数的时候忽略
		embed = self.zh_embed(src)
		embed = self.position_encoding(embed)

		# embed.shape [batch_size, seq_len, dim_model]
		memory = self.transformer.encoder(src=embed, src_key_padding_mask=src_pad_mask)
		# memory.shape [batch_size, seq_len, dim_model]
		return memory



	def decode(self, tgt, memory, tgt_mask, memory_pad_mask):
		# memory.shape [batch_size, seq_len, dim_model]
		# memory_pad_mask.shape [batch_size, seq_len] -- 编码器输出，如果最初序列中有pad, 最终输出结果也有pad[没有计算自注意力相关分数加权，仍是pad]
		# tgt_mask.shape [batch_size, tgt_len] -- 解码器输入，掩码，防止解码看到当前位置之后的信息
		# tgt.shape [batch_size, tgt_len]
		embed = self.en_embed(tgt)
		embed = self.position_encoding(embed)
		# embed.shape [batch_size, tgt_len, dim_model]

		output = self.transformer.decoder(memory=memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_pad_mask, tgt=embed)
		# outputs.shape [batch_size, tgt_len, dim_model]
		outputs = self.linear(output)
		# outputs.shape [batch_size, tgt_len, en_vocab_size]
		return outputs