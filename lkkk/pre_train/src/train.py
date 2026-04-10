import torch
from tqdm import tqdm

from lkkk.pre_train.src import config
from lkkk.pre_train.src.classifymodel import classifymodel
from lkkk.pre_train.src.dataloader import get_dataloader
from torch.utils.tensorboard import SummaryWriter
import time

def train_one_epoch(model, loss_fn, optimizer, device, dataloader):
	model.train()

	total_loss = 0
	for batch in tqdm(dataloader, desc="训练"):
		# 字典表达式 {key expression: val expression for k,v in original.items()}
		inputs = {k : v.to(device) for k,v in batch.items()}
		labels = inputs.pop("labels").to(dtype=torch.float)
		# shape [batch_size]

		outputs = model(**inputs)
		# shape [batch_size]

		# BCE loss -- 输出是一个值，要求output 和 target shape一致，都是【batch_size】
		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		total_loss += loss.item()
	return total_loss / len(dataloader)

def run_train():
	# 1. 设备
	device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	print(f"device: {device}")
	# 2. 数据获取
	data_loader = get_dataloader()
	# 3. 模型
	model = classifymodel().to(device)
	# 4. 损失函数
	loss_fn = torch.nn.BCEWithLogitsLoss()
	# 5. 优化器
	optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARN_RATE)
	# 6. Tensorboard Writer
	writer = SummaryWriter(log_dir=config.LOGS_DIR/time.strftime("%Y%m%d-%H%M%S"))

	bert_loss = float("inf")
	for epoch in range(1, config.EPOCHS+1):
		print(f"----Epoch {epoch}----")
		loss = train_one_epoch(model, loss_fn, optimizer, device, data_loader)
		print(f'loss: {loss}')

		writer.add_scalar('loss', loss, epoch)

		if loss < bert_loss:
			bert_loss = loss
			torch.save(model.state_dict(), config.MODELS_DIR/'best.pt')
			print("保存模型")

	writer.close()


if __name__ == '__main__':
	run_train()