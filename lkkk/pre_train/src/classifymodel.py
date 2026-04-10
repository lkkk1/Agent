from torch import nn
from transformers import AutoModel


class classifymodel(nn.Module):
	def __init__(self):
		super(classifymodel, self).__init__()
		self.bert = AutoModel.from_pretrained('bert-base-chinese')
		self.linear = nn.Linear(self.bert.config.hidden_size, 1)

	def forward(self, input_ids, attention_mask, token_type_ids):
		outputs = self.bert(input_ids, attention_mask, token_type_ids)

		last_hidden_state = outputs.last_hidden_state
		# shape [batch_size, seq_len, hidden_dim]

		cls_state = last_hidden_state[:,0,:]
		# shape [batch_size, hidden_dim]

		results = self.linear(cls_state).squeeze(-1)
		# shape [batch_size]
		return results
