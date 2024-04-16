import numpy as np

def print_training_dict(file_path):
    training_dict = np.load(file_path, allow_pickle=True).item()

    for key, value in training_dict.items():
        print(f"User ID: {key}, Items: {value}")

if __name__ == "__main__":
    # file_path = '../data/validation_dict.npy'
    file_path = '../data/category_feature.npy'
    # file_path = '../data/visual_feature.npy'
    # file_path = '../data/training_dict.npy'
    # file_path = '../data/testing_dict.npy'
    print_training_dict(file_path)
