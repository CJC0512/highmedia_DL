import gensim
import gensim.downloader as api
# FastText 모델 다운로드 (처음 실행 시 시간이 걸릴 수 있습니다)
model = api.load('fasttext-wiki-news-subwords-300')
# 두 단어 간 유사도 계산 함수
def word_similarity(word1, word2):
    try:
        similarity = model.similarity(word1, word2)
        return similarity
    except KeyError:
        return "하나 이상의 단어가 모델의 어휘에 없습니다."
# 사용 예시
word1 = "cat"
word2 = "dog"
similarity = word_similarity(word1, word2)
print(f"'{word1}'와 '{word2}' 사이의 유사도: {similarity}")
# 가장 유사한 단어 찾기
def most_similar_words(word, topn=5):
    try:
        similar_words = model.most_similar(word, topn=topn)
        return similar_words
    except KeyError:
        return "해당 단어가 모델의 어휘에 없습니다."
# 사용 예시
target_word = "computer"
similar_words = most_similar_words(target_word)
print(f"'{target_word}'와 가장 유사한 단어들:")
for word, score in similar_words:
    print(f"{word}: {score}")