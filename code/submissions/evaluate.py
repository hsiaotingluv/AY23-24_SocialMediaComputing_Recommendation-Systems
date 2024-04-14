import numpy as np
import torch

def evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, weight, visual_features, category_features, use_content_features, print_reco=True):
	recommends = []
	for i in range(len(top_k)):
		recommends.append([]) # Initialising Recommendation Lists

	with torch.no_grad(): # Disabling Gradient Computation
		pred_list_all = []
		for i in gt_dict.keys(): 
			if len(gt_dict[i]) != 0: # if user has ground truth items
				user = torch.full((item_num,), i, dtype=torch.int64).to(args.device) # create n_item users for prediction
				item = torch.arange(0, item_num, dtype=torch.int64).to(args.device) 

				# Fetch visual and category features for each item for ContentBasedMF
				if use_content_features:
					visual_feat = torch.FloatTensor(np.stack([visual_features[i] for i in range(item_num)])).to(args.device)
					category_feat = torch.LongTensor([category_features[i] for i in range(item_num)]).to(args.device)
					prediction = model(user, item, weight, visual_feat, category_feat)
					
				# MF Model
				else:
					prediction = model(user, item)
		
				prediction = prediction.detach().cpu().numpy().tolist()

				# Masking Known Interactions
				for j in train_dict[i]: # mask train
					prediction[j] -= float('inf')
				if flag == 1: # mask validation
					if i in valid_dict:
						for j in valid_dict[i]:
							prediction[j] -= float('inf')

				# Aggregating Predictions
				pred_list_all.append(prediction)

		# Converting Predictions to Tensor
		predictions = torch.Tensor(pred_list_all).to(args.device) 
		
		# Extracting Top-K Recommendations
		for idx in range(len(top_k)):
			_, indices = torch.topk(predictions, int(top_k[idx]))
			recommends[idx].extend(indices.tolist())
			# Print out the recommendations for this user
			if print_reco:
				for user_id in range(len(recommends[idx])):
					print(f"User {user_id} recommendations: {recommends[idx][user_id][:]}")

	return recommends

# Recall & Normalized Discounted Cumulative Gain (NDCG) - discounted version of recall, related to rank (higher the better)
def metrics(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, weight, visual_features, category_features, use_content_features, print_reco=False):
	# RECALL, NDCG = [], []
	RECALL, NDCG, ILD, ILD_VC, ITEM_COVERAGE, CATEGORY_COVERAGE = [], [], [], [], [], []
	recommends = evaluate(args, model, top_k, train_dict, gt_dict, valid_dict, item_num, flag, weight, visual_features, category_features, use_content_features, print_reco)

	for idx in range(len(top_k)):
		sumForRecall, sumForNDCG, user_length = 0, 0, 0
		k=-1
		for i in gt_dict.keys(): # for each user
			k += 1
			if len(gt_dict[i]) != 0:
				userhit = 0
				dcg = 0
				idcg = 0
				idcgCount = len(gt_dict[i])
				ndcg = 0

				for index, thing in enumerate(recommends[idx][k]):
					if thing in gt_dict[i]:
						userhit += 1
						dcg += 1.0 / (np.log2(index+2))
					if idcgCount > 0:
						idcg += 1.0 / (np.log2(index+2))
						idcgCount -= 1
				if (idcg != 0):
					ndcg += (dcg / idcg)

				sumForRecall += userhit / len(gt_dict[i])
				sumForNDCG += ndcg
				user_length += 1

		RECALL.append(round(sumForRecall/user_length, 4))
		NDCG.append(round(sumForNDCG/user_length, 4))

		for idx, rec_list in enumerate(recommends):
        # Calculate Intra-List Diversity and Item Coverage for the current top_k recommendation lists
			ild_vc = intra_list_diversity_with_visual_category(rec_list, visual_features, category_features, use_content_features)
			ild = intra_list_diversity(rec_list, category_features)
			item_cov = item_coverage(rec_list, item_num)
			category_cov = category_coverage(rec_list, category_features)

			ILD.append(ild)
			ILD_VC.append(ild_vc)
			ITEM_COVERAGE.append(item_cov)
			CATEGORY_COVERAGE.append(category_cov)

	return RECALL, NDCG, ILD, ILD_VC, ITEM_COVERAGE, CATEGORY_COVERAGE

def print_results(loss, valid_result, test_result, valid_ild, valid_ild_vc, valid_item_coverage, valid_category_coverage, valid_f1, test_ild, test_ild_vc, test_item_coverage, test_category_coverage, test_f1):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None:
        print("[Valid]: Recall: {} NDCG: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]])))
        if valid_ild is not None and valid_ild_vc is not None and valid_item_coverage is not None and valid_category_coverage is not None:
            print("[Valid]: ILD: {:.4f} ILD (Visual & Category): {:.4f} Item Coverage: {:.4f} Category Coverage: {:.4f} F1: {:.4f}".format(valid_ild, valid_ild_vc, valid_item_coverage, valid_category_coverage, valid_f1))
		
    if test_result is not None: 
        print("[Test]: Recall: {} NDCG: {} ".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]])))
        if test_ild is not None and test_ild_vc is not None and test_item_coverage is not None and test_category_coverage is not None:
            print("[Test]: ILD: {:.4f} ILD (Visual & Category): {:.4f} Item Coverage: {:.4f} Category Coverage: {:.4f} F1: {:.4f}".format(test_ild, test_ild_vc, test_item_coverage, test_category_coverage, test_f1))

# Calculate dissimilarity between two visual feature vectors
def visual_dissimilarity(vec_i, vec_j):
    return 1 - np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))

# Calculate dissimilarity between two category IDs (0 for same, 1 for different)
def category_dissimilarity(cat_i, cat_j):
    return 0 if cat_i == cat_j else 1

# Calculate Intra-List Diversity (ILD)
# Calculates diversity within each recommendation list and averages across all provided lists
def intra_list_diversity_with_visual_category(recommends, visual_features, category_features, use_content_features, visual_weight=0.5):
	if not use_content_features:
        # For MF model, calculate diversity based on categories alone
		total_category_diversity = 0
		for rec_list in recommends:
			categories_in_list = set(category_features[item] for item in rec_list)
            # Diversity is the proportion of unique categories in the recommendation list
			list_diversity = len(categories_in_list) / len(rec_list)
			total_category_diversity += list_diversity
			
		return total_category_diversity / len(recommends) if len(recommends) > 0 else 0
	
	# ContentBasedMF Model
	total_diversity = 0
	num_lists = len(recommends)

	for rec_list in recommends:
		list_diversity = 0
		num_pairs = 0

		for i in range(len(rec_list)):
			for j in range(i + 1, len(rec_list)):
				# Visual dissimilarity
				vis_dissimilarity = visual_dissimilarity(visual_features[rec_list[i]], visual_features[rec_list[j]])
				
				# Category dissimilarity
				cat_dissimilarity = category_dissimilarity(category_features[rec_list[i]], category_features[rec_list[j]])
				
				# Combined dissimilarity (weighted average)
				combined_dissimilarity = (visual_weight * vis_dissimilarity) + ((1 - visual_weight) * cat_dissimilarity)
				
				list_diversity += combined_dissimilarity
				num_pairs += 1

		if num_pairs > 0:
			total_diversity += list_diversity / num_pairs

	return total_diversity / num_lists if num_lists > 0 else 0

def intra_list_diversity(recommends, category_features):
    K = 10  # Top K items to consider for diversity calculation
    total_diversity = 0

    for rec_list in recommends:
        diversity_count = 0
        # Considering all unique pairs in the top K recommendations
        for i in range(K):
            for j in range(i + 1, K):
                # Increment diversity count if the categories are different
                if category_features[rec_list[i]] != category_features[rec_list[j]]:
                    diversity_count += 1
        
        # Compute diversity for the current list 
        list_diversity = (2 * diversity_count) / (K * (K - 1))
        total_diversity += list_diversity

    # Average diversity across all recommendation lists
    avg_diversity = total_diversity / len(recommends) if recommends else 0
    return avg_diversity

# Aggregate Unique Recommendations
# Long-Tail Exploration & Avoiding Concentration with popular items
# higher result -> content discovery, uncovering and recommending "long-tail" items (less popular but highly relevant)
# lower result ->  "rich get richer" phenomenon, popular items dominate recommendations
def item_coverage(recommends, num_total_items):
	# Get all unique items from all recommended item lists
    unique_items = set(item for rec_list in recommends for item in rec_list)
    item_coverage = len(unique_items) / num_total_items 
    return item_coverage

def category_coverage(recommends, category_features):
    # Track unique categories in the recommended items
    unique_categories = set()
    
    for rec_list in recommends:
        for item_id in rec_list:
            # Add the category of each recommended item to the set of unique categories
            unique_categories.add(category_features[item_id])
    
    # Calculate the category coverage
    category_coverage = len(unique_categories) / 368
    
    return category_coverage
