def calculate_accuracy(outputs, labels):
    import torch
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).float()
    return correct.sum() / len(labels)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compare_embeddings(embedding1, embedding2):
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
