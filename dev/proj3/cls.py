# STEP 1
from transformers import pipeline

# STEP 2
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
# STEP 3
# text = "현대바이오, '폴리탁셀' 코로나19 치료 가능성에 19% 급등" # positive
text = "늦었다고 생각할 땐 너무 늦은 거다. 그러니 지금 당장 시작해라" # neutral

# positive: 샤오미의 입장에서 해석됨. 반면, 삼성전자의 입장인 경우 negative로 해석되어야 함.
# text = "샤오미의 폴더블 폰의 점유율이 삼성전자 보다 높아졌다." 

# STEP 4
result = classifier(text)

# STEP 5
print(result)