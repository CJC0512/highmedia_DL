# STEP 1
from sentence_transformers import SentenceTransformer

# STEP 2
# 1. Load a pretrained Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# STEP 3
# The sentences to encode
sentence1 = "으하하하하하 대박났다"
sentence2 = "평생 놀고 먹고 살 수 있는 형편이지"

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# STEP 4
# 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings.shape)
# [3, 384]

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

# STEP 5
# 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
similarities = model.similarity(embedding1, embedding2)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])