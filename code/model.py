import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=8, dropout=0, mean=0):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)
        self.dropout = nn.Dropout(dropout)

        self.num_user = num_users

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()  # User bias
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()  # Item bias
        return self.dropout((U * I).sum(1) + b_u + b_i + self.mean) # self.mean is global bias, dropout to prevent overfitting
    
class ContentBasedMF(nn.Module):
    def __init__(self, num_users, num_items, visual_feature_size=512, category_size=368, embedding_size=8, dropout=0, mean=0):
        super(ContentBasedMF, self).__init__()

        # User embeddings and biases
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        
        # Item embeddings and biases
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        # Non-linear transformation for visual features
        self.visual_emb = nn.Sequential(
            nn.Linear(visual_feature_size, embedding_size * 2),
            nn.ReLU(),
            nn.Linear(embedding_size * 2, embedding_size),
        )
        self.visual_bias = nn.Parameter(torch.zeros(1))

        self.category_emb = nn.Embedding(category_size, embedding_size)
        self.category_bias = nn.Embedding(category_size, 1)

        # Initialization of weights and biases
        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        for layer in self.visual_emb:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, 0, 0.005)
        self.visual_bias.data.uniform_(-0.01, 0.01)

        self.category_emb.weight.data.uniform_(0, 0.005)
        self.category_bias.weight.data.uniform_(-0.01, 0.01)

        # Mean rating for initialization and dropout for regularization
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u_id, i_id, weight, visual_features, category_features):
        # Get user and item embeddings
        U = self.user_emb(u_id)
        I = self.item_emb(i_id)

        # Get biases
        b_u = self.user_bias(u_id).squeeze()      # User bias
        b_i = self.item_bias(i_id).squeeze()      # Item bias
        b_c = self.category_bias(category_features).squeeze()  # Category bias
        
        # Get content feature embeddings
        V = self.visual_emb(visual_features)        # Visual feature embeddings
        C = self.category_emb(category_features)    # Category feature embeddings

        I = (1 - weight) * I + weight * (V + C)
        prediction = ((U * I).sum(1) + b_u + b_i + weight * (self.visual_bias + b_c) + self.mean)

        return self.dropout(prediction)
    