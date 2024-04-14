import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import model
import evaluate
import data_utils

import random
import numpy as np

from torch.utils.tensorboard import SummaryWriter

def calculate_f1(ndcg, ild):
    return 2 * (ndcg * ild) / (ndcg + ild) if (ndcg + ild) > 0 else 0

if __name__ == "__main__":
	seed = 4242
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  # for multi-GPU
	torch.backends.cudnn.deterministic = True
	cudnn.benchmark = True

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", type=str, default="../data/", help="path for dataset")
	parser.add_argument("--model", type=str, default="MF", help="model name")
	parser.add_argument("--emb_size", type=int,default=64, help="predictive factors numbers in the model")
	parser.add_argument("--weight", type=float, default=0.5, help="weight of visual and category features")

	parser.add_argument("--visual_feature_size", type=int, default=512, help="size of the visual feature vector")
	parser.add_argument("--category_size", type=int, default=368, help="number of unique categories")

	parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
	parser.add_argument("--dropout", type=float,default=0.0,  help="dropout rate")
	parser.add_argument("--batch_size", type=int, default=1024, help="batch size for training")
	parser.add_argument("--epochs", type=int, default=100, help="training epoches")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on. 'cuda' for GPU or 'cpu' for CPU.")

	parser.add_argument("--top_k", default='[10]', help="compute metrics@top_k")
	parser.add_argument("--log_name", type=str, default='log', help="log_name")
	parser.add_argument("--model_path", type=str, default="./models/", help="main path for model")

	parser.add_argument("--gpus", default="0", type=str, help="Comma separated list of GPU IDs to be used (e.g. '0,1,2,3')")

	args = parser.parse_args()

	# Setting up GPUs
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
	device = torch.device(args.device)
	gpus = [int(gpu) for gpu in args.gpus.split(',') if gpu.strip()]

	# Initialize SummaryWriter
	writer = SummaryWriter("runs/{}BestRecall_NDCG_{}_{}_{}lr_{}dr".format(args.model_path, args.weight, args.model, args.lr, args.dropout))

	# Early Stopping Initialization
	patience = 10
	best_recall = 0.0
	best_NDCG = 0.0
	best_f1 = 0.0
	no_improvement_count = 0

	############################ PREPARE DATASET ##########################
	train_path = args.data_path + '/training_dict.npy'
	valid_path = args.data_path + '/validation_dict.npy'
	test_path = args.data_path + '/testing_dict.npy'
	category_path = args.data_path + '/category_feature.npy'
	visual_path = args.data_path + '/visual_feature.npy'

	# Load data and features
	user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt, visual_features, category_features = data_utils.load_all(train_path, valid_path, test_path, category_path, visual_path)

	# Construct the train datasets & dataloader
	train_dataset = data_utils.MFData(train_data, item_num, train_dict, True)
	train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

	########################### CREATE MODEL ##############################
	if args.model == 'MF':
		model = model.MF(user_num, item_num, args.emb_size, args.dropout)
	if args.model == 'ContentBasedMF':
		model = model.ContentBasedMF(user_num, item_num, args.visual_feature_size, args.category_size, args.emb_size, args.dropout)
    
	if torch.cuda.device_count() > 1 and len(gpus) > 1:
		print(f"Using {torch.cuda.device_count()} GPUs!")
		model = nn.DataParallel(model, device_ids=gpus)

	model.to(args.device)
	loss_function = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	########################### TRAINING ##################################
	for epoch in range(args.epochs):
		# train
		model.train() # Enable dropout (if have).
		start_time = time.time()
		train_loader.dataset.ng_sample()

		for user, item, label in train_loader:
			user = user.to(args.device)
			item = item.to(args.device)
			label = label.float().to(args.device)

			model.zero_grad()

			if args.model == 'ContentBasedMF':
				# Fetch visual and category features for the batch items
				visual_feat = torch.FloatTensor(np.stack([visual_features[i.item()] for i in item])).to(args.device)
				category_feat = torch.LongTensor([category_features[i.item()] for i in item]).to(args.device)
				prediction = model(user, item, args.weight, visual_feat, category_feat)

			# MF model
			else:
				prediction = model(user, item)

			loss = loss_function(prediction, label)
			loss.backward()
			optimizer.step()
		
		if (epoch+1) % 1 == 0: # always true
			model.eval() # evaluation
	
			use_content_features = (args.model == 'ContentBasedMF')

			valid_recall, valid_ndcg, valid_ild, valid_ild_vc, valid_item_coverage, valid_category_coverage = evaluate.metrics(args, model, eval(args.top_k), train_dict, valid_dict, valid_dict, item_num, 0, args.weight, visual_features, category_features, use_content_features)

			valid_f1 = calculate_f1(valid_ndcg[0], valid_ild[0])

			test_recall, test_ndcg, test_ild, test_ild_vc, test_item_coverage, test_category_coverage = evaluate.metrics(args, model, eval(args.top_k), train_dict, test_dict, valid_dict, item_num, 1, args.weight, visual_features, category_features, use_content_features)

			test_f1 = calculate_f1(test_ndcg[0], test_ild[0])

			elapsed_time = time.time() - start_time

			# Early Stopping Check based on F1 Score on valid set
			if valid_recall[0] > best_recall:
			# if valid_f1 > best_f1:
			# if valid_ndcg[0] > best_NDCG:
				# best_NDCG = valid_ndcg[0]
				best_recall = valid_recall[0]
				# best_f1 = valid_f1
				no_improvement_count = 0
				best_epoch = epoch
				print('---'*18)
				# print(f"Highest NDCG: {best_NDCG} at epoch: {best_epoch}")
				print(f"Highest Recall: {best_recall} at epoch: {best_epoch}")
				# print(f"Highest F1: {best_f1} at epoch: {best_epoch}")
				print('---'*18)
				best_results = (valid_recall, valid_ndcg)
				best_test_results = (test_recall, test_ndcg)
				best_valid_ild = valid_ild[0]
				best_valid_ild_vc = valid_ild_vc[0]
				best_valid_item_coverage = valid_item_coverage[0]
				best_valid_category_coverage = valid_category_coverage[0]
				best_valid_f1 = valid_f1
				best_test_ild = test_ild[0]
				best_test_ild_vc = test_ild_vc[0]
				best_test_item_coverage = test_item_coverage[0]
				best_test_category_coverage = test_category_coverage[0]
				best_test_f1 = test_f1

				# save model
				if not os.path.exists(args.model_path):
					os.mkdir(args.model_path)
				torch.save(model, '{}BestRecall_NDCG_{}_{}_{}lr_{}dr.pth'.format(args.model_path, args.weight, args.model, args.lr, args.dropout))

			else:
				no_improvement_count += 1
				if no_improvement_count >= patience:
					print(f"No improvement in Recall for {patience} consecutive epochs. Stopping early.")
					break

			# Logging metrics to TensorBoard
			writer.add_scalars('Recall', {'Valid': valid_recall[0], 'Test': test_recall[0]}, epoch)
			writer.add_scalars('NDCG', {'Valid': valid_ndcg[0], 'Test': test_ndcg[0]}, epoch)
			writer.add_scalars('ILD', {'Valid': valid_ild[0], 'Test': test_ild[0]}, epoch)
			writer.add_scalars('ILD_Visual_Category', {'Valid': valid_ild_vc[0], 'Test': test_ild_vc[0]}, epoch)
			writer.add_scalars('Item_Coverage', {'Valid': valid_item_coverage[0], 'Test': test_item_coverage[0]}, epoch)
			writer.add_scalars('Category_Coverage', {'Valid': valid_category_coverage[0], 'Test': test_category_coverage[0]}, epoch)
			writer.add_scalars('F1_Score', {'Valid': valid_f1, 'Test': test_f1}, epoch)

			print('---'*18)
			print("The time elapse of epoch {:03d}".format(epoch) + " is: " +  time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
			evaluate.print_results(None, (valid_recall, valid_ndcg), (test_recall, test_ndcg), valid_ild[0], valid_ild_vc[0], valid_item_coverage[0], valid_category_coverage[0], valid_f1, test_ild[0], test_ild_vc[0],test_item_coverage[0], test_category_coverage[0], test_f1)
			print('---'*18)
				
	print('==='*18)
	print(f"End. Best Epoch with highest Recall on valid set is {best_epoch}")
	evaluate.print_results(None, best_results, best_test_results, best_valid_ild, best_valid_ild_vc, best_valid_item_coverage, best_valid_category_coverage, best_valid_f1, best_test_ild, best_test_ild_vc, best_test_item_coverage, best_test_category_coverage, best_test_f1)
	writer.close()
	print('Training completed.')