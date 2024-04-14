import argparse
import torch
import torch.backends.cudnn as cudnn

import model
import evaluate
import data_utils

import random
import numpy as np

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
    test_path = args.data_path + '/testing_dict.npy'
    category_path = args.data_path + '/category_feature.npy'
    visual_path = args.data_path + '/visual_feature.npy'
    # test_path = args.data_path + '/heldout_dict.npy' # for live evaluation

    use_content_features = (args.model == 'ContentBasedMF')

    user_num, item_num, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt, visual_features, category_features = data_utils.load_all(train_path, valid_path, test_path, category_path, visual_path)

    ########################### LOAD MODEL #################################
    model = torch.load(f"./models/{args.ckpt}")
    model.to(args.device)

    ########################### EVALUATION #####################################
    model.eval()

    test_recall, test_ndcg, test_ild, test_ild_vc, test_item_coverage, test_category_coverage = evaluate.metrics(args, model, eval(args.top_k), train_dict, test_dict, valid_dict, item_num, 1, 0.1, visual_features, category_features, use_content_features, print_reco=True)

    test_f1 = calculate_f1(test_ndcg[0], test_ild[0])

    print('---'*18)
    evaluate.print_results(None, None, (test_recall, test_ndcg), None, None, None, None, None, test_ild[0], test_ild_vc[0], test_item_coverage[0], test_category_coverage[0], test_f1) 
    print('---'*18)