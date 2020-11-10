from model import VGG
from dataset import DatasetForClassifier
import argparse
from torchvision import transforms
import torch
import torch.optim as optim
from tqdm import tqdm
from torch import nn
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='VGG Pretrain')

	parser.add_argument('--resolution', type = int)
	parser.add_argument('--dataroot', type = str)
	parser.add_argument('--val_split', type = float, default = 0.2)
	parser.add_argument('--lr', type = float, default = 0.001)
	parser.add_argument('--batch_size', type = int, default = 64)
	parser.add_argument('--save_path', type = str, default = 'vgg_weights')
	parser.add_argument('--save_prefix', type = str, default = 'VGG_13')

	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	res_to_class = {8 : 20, 16 : 40, 32 : 80, 64 : 160, 128 : 320, 256 : None}
	args.classes = res_to_class[args.resolution]

	res_to_epochs = {8 : 30, 16 : 35, 32 : 40, 64 : 40, 128 : 45, 256 : 50}
	args.epochs = res_to_epochs[args.resolution]

	transform = transforms.Compose(
        [
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

	dataset = DatasetForClassifier(args.dataroot, transform, resolution = args.resolution, classes = args.classes, val_percent = args.val_split)
	dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

	args.classes = dataset.total_ids

	model = VGG(args.resolution, args.classes).to(device)
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	criterion = nn.CrossEntropyLoss()



	best_val_acc = 0
	for epoch in range(args.epochs):
		print("-" * 15)
		print(f"Epoch {epoch}/{args.epochs} started")

		model.train()
		dataset.train()

		pbar = tqdm(range(len(dataset_loader)))
		pbar_iter = iter(pbar)


		running_train_acc = 0
		running_train_loss = 0
		for i, data in enumerate(dataset_loader):
			images = data['img'].to(device)
			labels = data['img_id'].to(device)

			optimizer.zero_grad()

			pred_labels = model(images)
			loss = criterion(pred_labels, labels)
			loss.backward()

			optimizer.step()

			train_loss = loss.item()
			_, pred_labels_ids = torch.max(pred_labels, 1)
			train_acc = (pred_labels_ids == labels).sum().item() / labels.size(0)

			running_train_loss += train_loss
			running_train_acc += train_acc
			if i % 10 == 0:
				pbar.set_description(f'Train Loss: {(running_train_loss / 10):.4f}; Train Accuracy: {(running_train_acc / 10):.4f}')
				running_train_acc = 0
				running_train_loss = 0

			next(pbar_iter)

		pbar.close()
		model.eval()
		dataset.eval()
		number_of_right = 0
		with torch.no_grad():
			for data in dataset_loader:
				images = data['img'].to(device)
				labels = data['img_id'].to(device)

				pred_labels = model(images)

				_, pred_labels_ids = torch.max(pred_labels, 1)
				number_of_right += (pred_labels_ids == labels).sum().item()


			val_acc = number_of_right / len(dataset)
			print()
			print(f"Val Accuracy: {val_acc:.4f}")
			if val_acc > best_val_acc:
				model.save(args.save_path, args.save_prefix)
				with open(os.path.join(args.save_path, f'{args.save_prefix}_{args.resolution}.txt'), 'w') as f:
					f.write(f'{val_acc:.4f}\n')
				best_val_acc = val_acc



	# model.save(args.save_path)







