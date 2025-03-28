import numpy as np 
import torch.utils.data as data


def load_all(train_path, valid_path, test_path, category_feature_path, visual_feature_path):
	""" We load all the three file here to save time in each epoch. """
	train_dict = np.load(train_path, allow_pickle=True).item()
	valid_dict = np.load(valid_path, allow_pickle=True).item()
	test_dict = np.load(test_path, allow_pickle=True).item()

	# Load visual and category features
	visual_features = np.load(visual_feature_path, allow_pickle=True).item()
	category_features = np.load(category_feature_path, allow_pickle=True).item()

	user_num, item_num = 0, 0
	user_num = max(user_num, max(train_dict.keys()))
	user_num = max(user_num, max(valid_dict.keys()))
	user_num = max(user_num, max(test_dict.keys()))
	
	train_data, valid_gt, test_gt = [], [], []
	for user, items in train_dict.items():
		item_num = max(item_num, max(items))
		for item in items:
			train_data.append([int(user), int(item)])
	for user, items in valid_dict.items():
		item_num = max(item_num, max(items))
		for item in items:
			valid_gt.append([int(user), int(item)])
	for user, items in test_dict.items():
		item_num = max(item_num, max(items))
		for item in items:
			test_gt.append([int(user), int(item)])

	return user_num+1, item_num+1, train_dict, valid_dict, test_dict, train_data, valid_gt, test_gt, visual_features, category_features

class MFData(data.Dataset):
	def __init__(self, features, num_item, train_dict=None, is_training=None):
		super(MFData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features_ps = features
		self.num_item = num_item
		self.train_dict = train_dict
		self.is_training = is_training
		self.labels = [0 for _ in range(len(features))]

	# Negative sampling
	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'

		self.features_ng = []
		for x in self.features_ps:
			u = x[0] # User ID from positive sample
			j = np.random.randint(self.num_item) # Randomly selects an item ID as a potential negative sample
			while j in self.train_dict[u]:
				j = np.random.randint(self.num_item)
			self.features_ng.append([u, j])

		labels_ps = [1 for _ in range(len(self.features_ps))] # Indicate positive user-item interaction
		labels_ng = [0 for _ in range(len(self.features_ng))] # Indicate negative user-item interaction

		self.features_fill = self.features_ps + self.features_ng # Combined feature list
		self.labels_fill = labels_ps + labels_ng # Combined labels

	def __len__(self):
		return (1 + 1) * len(self.labels)

	def __getitem__(self, idx):
		features = self.features_fill if self.is_training else self.features_ps
		labels = self.labels_fill if self.is_training else self.labels

		user = features[idx][0]
		item = features[idx][1]
		label = labels[idx]
		return user, item ,label