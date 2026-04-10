from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
from tqdm import tqdm

class BaseTokenizer:
	unknow_token = '<unk>'
	padding_token = '<pad>'
	sos_token = '<sos>'
	eos_token = '<eos>'

	def __init__(self, vocab_list):
		self.vocab_list = vocab_list
		self.vocab_size = len(self.vocab_list)
		self.word2index = {word : index for index, word in enumerate(self.vocab_list)}
		self.index2word = {index : word for index, word in enumerate(self.vocab_list)}
		self.unknown_token_index = self.word2index[self.unknow_token]
		self.padding_token_index = self.word2index[self.padding_token]
		self.sos_token_index = self.word2index[self.sos_token]
		self.eos_index = self.word2index[self.eos_token]

	@classmethod
	def tokenize(cls, sentence) -> list[str]:
		pass

	@classmethod
	def build_vocab_list(cls, sentences, vocab_path):
		with (open(vocab_path) as f):
			vocab_set = set()
			for sentence in tqdm(sentences, desc="构建词表"):
				vocab_set.update(cls.tokenize(sentence))

			vocab_list = [cls.padding_token, cls.unknow_token, cls.sos_token, cls.eos_token]+[token for token in vocab_set if token.strip() != '']
			with open(vocab_path, "w", encoding="utf-8") as f:
				f.write("\n".join(vocab_list))

	@classmethod
	def from_vocab(cls, vocab_path):
		with open(vocab_path, "r", encoding="utf-8") as f:
			vocab_list = [line.strip() for line in f.readlines()]
		return cls(vocab_list)

	def encode(self, sentence, add_sos_eos=False):
		tokens = self.tokenize(sentence)
		if add_sos_eos:
			tokens = [self.sos_token] + tokens + [self.eos_token]
		return [self.word2index.get(token, self.unknown_token_index) for token in tokens]

class EnglishTokenizer(BaseTokenizer):
	# nltk -- 英文分词器TreebankWordTokenizer，可以拆分缩写，处理标点，
	# 且有对应的TreebankWordDetokenizer 可以按encode相反逻辑，恢复缩写和标点处理，还原句子
	tokenizer = TreebankWordTokenizer()
	detokenizer = TreebankWordDetokenizer()

	def __init__(self, vocab_list):
		super().__init__(vocab_list)

	@classmethod
	def tokenize(cls, sentence):
		return cls.tokenizer.tokenize(sentence)

	def decode(self, indexes):
		tokens = [self.index2word[index] for index in indexes]
		return self.detokenizer.detokenize(tokens)

class ChineseTokenizer(BaseTokenizer):
	def __init__(self, vocab_list):
		super().__init__(vocab_list)

	@classmethod
	def tokenize(cls, sentence):
		return list(sentence)