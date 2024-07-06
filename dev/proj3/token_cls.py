# STEP 1
from transformers import pipeline

# STEP 2
# classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
classifier = pipeline("ner", model="Leo97/KoELECTRA-small-v3-modu-ner")


# STEP 3
# text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
text = "서울역을 거쳐 김포공항역으로 가는 공항철도에 대해서 안내해줘. 추가적으로 홍대입구역, 마곡나루역에 대해서 설명해봐"

# STEP 4
result = classifier(text)

# STEP 5
print(result)