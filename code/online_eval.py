import argparse
import torch
import torch.backends.cudnn as cudnn

import model
import evaluate
import data_utils

import numpy as np
import time

'''
    There are several TODOs for you to check or modify for online evaluation. All todo are marked as "TODO: XXX"
'''

# function definition
def metrics_f1(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag):
	'''
		The input parameters are the same as in previously given function of ``metrics`` in evaluate.py. 
	'''
	from itertools import combinations

	NDCG, F1_ndcg = [], []
	ILD = []
	
    # TODO: Make sure you can load ``category_feature.npy`` from your path.
	category_features = np.load("../data/category_feature.npy", allow_pickle=True).item() # load the category file. 
	visual_features = np.load("../data/visual_feature.npy", allow_pickle=True).item() # load the visual file. 

	# TODO: You can modify this part for your model inference. Make sure it returns a list of recommendations for each user (refer to the previously given codes)
	recommends = evaluate.evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, 0.1, visual_features, category_features, True, True)

	# F1 metrics. Do NOT modify this part!
	for idx in range(len(top_k)):
		sumForNDCG, sumForILD, sumForF1_ndcg, user_length = 0, 0, 0, 0
		k=-1
		for i in gt_dict.keys(): # for each user
			k += 1
			if len(gt_dict[i]) != 0:
				dcg = 0
				idcg = 0
				idcgCount = len(gt_dict[i])
				ndcg = 0

				for index, thing in enumerate(recommends[idx][k]):
					if thing in gt_dict[i]:
						dcg += 1.0 / (np.log2(index+2))
					if idcgCount > 0:
						idcg += 1.0 / (np.log2(index+2))
						idcgCount -= 1
				if (idcg != 0):
					ndcg += (dcg / idcg)

				# category ILD
				# get the category feature list
				cat_lst = [category_features[thing] for thing in recommends[idx][k]]

				# calculate the sum of 1s for non-equal pairs
				sum_diff = sum(1 for x, y in combinations(cat_lst, 2) if x != y)
				
				# normalize the sum
				ild = 2 * sum_diff / (top_k[idx] * (top_k[idx]-1))
				
				# F1
				f1_ndcg = 2 * ndcg * ild / (ndcg + ild)

				sumForNDCG += ndcg
				sumForILD += ild
				sumForF1_ndcg += f1_ndcg
				user_length += 1

		NDCG.append(round(sumForNDCG/user_length, 4))
		ILD.append(round(sumForILD/user_length, 4))
		F1_ndcg.append(round(sumForF1_ndcg/user_length, 4))

	return NDCG, ILD, F1_ndcg


def print_results(loss, valid_result, test_result):
    """output the evaluation results. No need to modify this part."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: NDCG: {} ILD: {} F1_NDCG: {} ".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]),
							'-'.join([str(x) for x in valid_result[2]])))
    if test_result is not None: 
        print("[Test]: NDCG: {} ILD: {} F1_NDCG: {} ".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]),
							'-'.join([str(x) for x in test_result[2]])))

if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", default='[10]', help="compute metrics@top_k")
    parser.add_argument("--data_path", type=str, default="../data/", help="main path for dataset")
    parser.add_argument("--model", type=str, default="MF", help="model name")
    parser.add_argument("--ckpt", type=str, default="MF_0.001lr_64emb_log.pth")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    cudnn.benchmark = True

    ############################## PREPARE DATASET ##########################
    train_path = args.data_path + '/training_dict.npy'
    valid_path = args.data_path + '/validation_dict.npy'
    category_path = args.data_path + '/category_feature.npy'
    visual_path = args.data_path + '/visual_feature.npy'
    test_path = args.data_path + '/heldout_dict.npy' # TODO: Make sure you can load the heldout test set during online evaluation.
	
    # test_path = args.data_path + '/testing_dict.npy' # TODO: You can uncomment this line to check the script by using testing_dict.npy.
    
    '''
        TODO: You should modify the dataset preparation part to run your model, e.g., load visual feature file or category feature file. 
    '''

    user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt, visual_features, category_features = data_utils.load_all(train_path, valid_path, test_path, category_path, visual_path)

    ########################### LOAD MODEL #################################
    '''
        TODO: You should set the args.ckpt to the path of your best model for online evaluation
    '''
    model = torch.load(f"./models/{args.ckpt}") 
    model.to(args.device)

    ########################### EVALUATION #####################################
    model.eval()
    t_start = time.time()
    test_result = metrics_f1(args, model, eval(args.top_k), train_dict, test_dict, valid_dict, item_num, 1)

    print('---'*18)
    print_results(None, None, test_result) 
    print('---'*18)
    print(f"The inference costs {time.time()-t_start} seconds.")