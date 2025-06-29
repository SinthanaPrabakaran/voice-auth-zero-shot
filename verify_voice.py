import numpy as np
from extract_embedding import extract_embedding
from utils import compare_embeddings

def verify_user(voice_path, enrolled_path):
    new_embedding = extract_embedding(voice_path)
    enrolled_embedding = np.load(enrolled_path)

    similarity = compare_embeddings(new_embedding, enrolled_embedding)
    print(f"Similarity Score: {similarity:.4f}")

    if similarity > 0.8:  # You can tune this threshold
        print("✅ Access Granted")
    else:
        print("❌ Access Denied")
