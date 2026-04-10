import torch
from transformers import AutoModelForSequenceClassification
from lkkk.pre_train_bertcls.src.dataloader import get_dataloader
from lkkk.pre_train_bertcls.src import config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(model, optimizer, data_loader, device):
	model.train()

	total_loss = 0
	for batch in tqdm(data_loader, desc="训练"):
		inputs = {k: v.to(device) for k,v in batch.items()}

		outputs = model(**inputs)
		loss = outputs.loss

		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		total_loss += loss.item()
	return total_loss / len(data_loader)

def run_train():
	# 设备
	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	# 模型
	model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese').to(device)
	# dataloader
	train_dataloader = get_dataloader()

	# 优化器
	optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARN_RATE)
	# 模型前向传播已经集成了loss 计算，不需要loss_fn

	# tensorboard
	writer = SummaryWriter(log_dir=config.LOGS_DIR)

	best_loss = float('inf')
	for epoch in range(1, config.EPOCHS + 1):
		print(f'Epoch {epoch}')
		loss = train_one_epoch(model, optimizer, train_dataloader, device)
		print(f'loss: {loss}')

		writer.add_scalar('loss', loss, epoch)

		if loss < best_loss:
			best_loss = loss
			model.save_pretrained(config.MODELS_DIR)
			print("保存模型")

	writer.close()

if __name__ == '__main__':
	run_train()