import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

documents = [
    "The quick brown fox jumps over the lazy dog",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
    "The five boxing wizards jump quickly",
    "How vexingly quick daft zebras jump!",
    "Bright vixens jump; dozy fowl quack"
]

def preprocess_document(doc):
    doc = doc.lower()  #conversion lowercase
    doc = re.sub(r'[^\w\s]', '', doc)  #remving punctuation
    return doc

processed_documents = [preprocess_document(doc) for doc in documents]

def generate_shingles(doc, k=3):
    shingles = set()
    for i in range(len(doc) - k + 1):
        shingle = doc[i:i+k]
        shingles.add(shingle)
    return shingles

shingle_sets = [generate_shingles(doc) for doc in processed_documents]

vocab = set.union(*shingle_sets)

def create_sparse_vector(shingles, vocab):
    sparse_vector = []
    for shingle in vocab:
        if shingle in shingles:
            sparse_vector.append(1)
        else:
            sparse_vector.append(0)
    return sparse_vector

sparse_vectors = [create_sparse_vector(shingles, vocab) for shingles in shingle_sets]

def generate_hash_func(size):
    hash_ex = list(range(1, size + 1))
    np.random.shuffle(hash_ex)
    return hash_ex

def compute_signature(vector, hash_functions):
    signature = []
    for hf in hash_functions:
        min_hash = float('inf')
        for i, val in enumerate(vector):
            if val == 1:
                hash_val = (i + 1) * hf[i] % (len(vector) + 1)
                if hash_val < min_hash:
                    min_hash = hash_val
        signature.append(min_hash)
    return signature

num_hash_functions = 20
hash_functions = [generate_hash_func(len(vocab)) for _ in range(num_hash_functions)]
signatures = [compute_signature(vector, hash_functions) for vector in sparse_vectors]

def split_into_bands(signature, b):
    assert len(signature) % b == 0
    r = int(len(signature) / b)
    subvecs = []
    for i in range(0, len(signature), r):
        subvecs.append(tuple(signature[i:i + r]))
    return subvecs

def find_candidate_pairs(signatures):
    candidate_pairs = []
    buckets = {}
    for idx, signature in enumerate(signatures):
        for band_idx, band in enumerate(split_into_bands(signature, b)):
            if band not in buckets:
                buckets[band] = [idx]
            else:
                for bucket_idx in buckets[band]:
                    candidate_pairs.append((idx, bucket_idx))
                buckets[band].append(idx)
    return candidate_pairs

b = 5  #no.of bands
r = 4   #no. of rows per band

candidate_pairs = find_candidate_pairs(signatures)

def jaccard_similarity(a, b):
    intersection = len(set(a).intersection(set(b)))
    union = len(set(a).union(set(b)))
    return intersection / union

original_jaccard_similarities = np.zeros((len(documents), len(documents)))
minhash_jaccard_similarities = np.zeros((len(documents), len(documents)))

for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        original_jaccard_similarities[i][j] = jaccard_similarity(shingle_sets[i], shingle_sets[j])
        minhash_jaccard_similarities[i][j] = jaccard_similarity(signatures[i], signatures[j])

print("Original Jaccard Similarities:")
print(original_jaccard_similarities)
print("\nMinhash Jaccard Similarities:")
print(minhash_jaccard_similarities)

plt.figure(figsize=(10, 6))
sns.heatmap(minhash_jaccard_similarities, annot=True, cmap="YlGnBu", xticklabels=documents, yticklabels=documents)
plt.title("Minhash Jaccard Similarities")
plt.xlabel("Documents")
plt.ylabel("Documents")
plt.show()
