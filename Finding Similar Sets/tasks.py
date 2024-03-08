from random import shuffle

def shingle(text, k):
    shingles = set()
    for i in range(len(text) - k + 1):
        shingle = text[i:i + k]
        shingles.add(shingle)
    return shingles

sentence_a = "flying fish flew by the space station"
sentence_b = "he will not allow you to bring your sticks of dynamite and pet armadillo along"
sentence_c = "he figured a few sticks of dynamite youre easier than a fishing pole to catch an armadillo"

k = 2

shingles_a = shingle(sentence_a, k)
shingles_b = shingle(sentence_b, k)
shingles_c = shingle(sentence_c, k)

shingle_vocabulary = shingles_a.union(shingles_b, shingles_c)

print("Shingles for sentence a:", shingles_a)
print("Shingles for sentence b:", shingles_b)
print("Shingles for sentence c:", shingles_c)
print("Shingle vocabulary:", shingle_vocabulary)


#Task 2
vocab = list(shingles_a.union(shingles_b).union(shingles_c))

a_1hot = [1 if x in shingles_a else 0 for x in vocab]
b_1hot = [1 if x in shingles_b else 0 for x in vocab]
c_1hot = [1 if x in shingles_c else 0 for x in vocab]

print("Sparse vector for sentence a:", a_1hot)


#Task 3
def create_hash_func(size: int):
    hash_ex = list(range(1, size + 1))
    shuffle(hash_ex)
    return hash_ex

def build_minhash_func(vocab_size, nbits):
    hashes = []
    for _ in range(nbits):
        hashes.append(create_hash_func(vocab_size))
    return hashes

def create_hash(vector: list):
    signature = []
    for func in minhash_func:
        for i in range(1, len(vocab) + 1):
            idx = func.index(i)
            signature_val = vector[idx]
            if signature_val == 1:
                signature.append(idx)
                break
    return signature

def jaccard(a, b):
    return len(a.intersection(b)) / len(a.union(b))

minhash_func = build_minhash_func(len(vocab), 20)

a_sig = create_hash(a_1hot)
b_sig = create_hash(b_1hot)
c_sig = create_hash(c_1hot)

print("Signature for document a:", a_sig)
print("Signature for document b:", b_sig)
print("Signature for document c:", c_sig)

similarity_ab = jaccard(set(a_sig), set(b_sig))
similarity_ac = jaccard(set(a_sig), set(c_sig))
similarity_bc = jaccard(set(b_sig), set(c_sig))

print("Jaccard similarity between a and b:", similarity_ab)
print("Jaccard similarity between a and c:", similarity_ac)
print("Jaccard similarity between b and c:", similarity_bc)


#Task 4
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def split_vector(signature, b):
    assert len(signature) % b == 0
    r = int(len(signature) / b)
    subvecs = []
    for i in range(0, len(signature), r):
        subvecs.append(signature[i:i + r])
    return subvecs

def probability(s, r, b):
    # s: similarity
    # r: rows (per band)
    # b: number of bands
    return 1 - (1 - s ** r) ** b

band_b = split_vector(b_sig, 10)
band_c = split_vector(c_sig, 10)
band_a = split_vector(a_sig, 10)

for b_rows, c_rows in zip(band_b, band_c):
    if b_rows == c_rows:
        print(f"Candidate pair: {b_rows} {c_rows}")
        # we only need one band to match
        break

for a_rows, c_rows in zip(band_a, band_c):
    if a_rows == c_rows:
        print(f"Candidate pair: {a_rows} {c_rows}")
        # we only need one band to match
        break

results = []
for s in np.arange(0.01, 1, 0.01):
    total = 100
    for b in [100, 50, 25, 20, 10, 5, 4, 2, 1]:
        r = int(total / b)
        P = probability(s, r, b)
        results.append({'s': s, 'P': P, 'r,b': f"{r}, {b}"})

results_df = pd.DataFrame(results)

sns.lineplot(data=results_df, x="s", y="P", hue="r,b")
plt.xlabel("Similarity")
plt.ylabel("Probability of Candidate Pair")
plt.title("Probability vs Similarity for Different Values of r and b")
plt.show()
