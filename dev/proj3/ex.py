# STEP 1
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration


# STEP 2
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# Load Model and Tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("EbanLee/kobart-summary-v3")
model = BartForConditionalGeneration.from_pretrained("EbanLee/kobart-summary-v3")

compare = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 3
title = "다음달부터 가스요금 오르는데...한전 '역대급 부채'에도 전기요금 동결" 

summary = "8월부터 민수용 도시가스 요금 6.8% 인상. 전기요금은 지난해 3분기 이후 5개 분기 연속 동결"

body = "정부가 8월부터 민수용 도시가스 요금을 6.8% 인상하기로 결정했다. 다만 고물가 장기화에 서민 부담을 우선 고려해야 한다는 판단에 따라 전기요금은 지난해 3·4분기 이후 5개 분기 연속 동결 상태다. \
        6일 에너지업계 등에 따르면 한전은 올 3·4분기(7~9월) 전기요금을 동결했다. 전기요금은 물가당국인 기획재정부와 에너지 주무부처인 산업통상자원부 협의를 거쳐 매년 3·6·9·12월 네 차례에 걸쳐 결정된다. \
        고물가가 장기화하고 있는 상황 속 서민 부담을 줄여야 한다는 판단이 작용했다. 사상 최악의 재무위기를 겪고 있는 한전으로서는 아쉬운 결정이라는 목소리가 나온다.\
        실제로 한전의 총부채는 201조원(지난 3월말 기준)에 육박하고, 지난해까지 누적적자(연결기준)만 42조원에 달하는 상황이다. 부채로 인해 매년 내야 할 이자만도 4조~5조 원에 육박할 것으로 전해졌다.\
        이런 가운데 가스공사는 지난 5일, 내달 1일부터 도시가스 주택용 도매요금을 MJ당 1.41원(서울 소매요금 기준 6.8%) 인상한다고 밝혔다. 일반용 도매요금은 MJ당 1.30원 인상될 예정이다. \
        인상 규모로 보면 서울시 4인 가구 기준 월 가스요금이 3770원 증가할 것이라는 예측이다. '에너지요금 현실화'가 불가피한 상황에서 서민 부담을 조금이나마 완화하기 위해 가스 사용량이 적은 하절기에 요금을 인상한 것으로 풀이된다.\
        현재 가스공사 역시 13조5000억 원에 달하는 미수금에 심각한 재정난을 겪고 있다. 미수금은 천연가스 수입 대금 중 가스요금으로 회수되지 않은 금액으로, 사실상의 영업손실액이다.\
        한전과 가스공사의 재무 악화가 심각한 상황에서 '생산 원가'에 영향을 미치는 불안한 국제유가 상황도 더 이상 요금 현실화를 미룰 수 없는 조건 중 하나다. \
        올 상반기 배럴당 80달러 선에 형성된 국제유가는 최근 중동 분쟁으로 상승하는 추세다.\
        지난 1일(현지시간) 뉴욕상업거래소에서 8월 인도분 서부텍사스산 원유(WTI)는 전일보다 1.84달러(2.3%) 상승한 83.38달러에 거래를 마쳤다.\
        이는 지난 4월 이후 두 달여 만에 가장 높은 수준이다.\
        국제유가 상승으로 인해 에너지공기업들의 생산원가가 오르면 이를 에너지 요금 인상분에 반영해야 하는데, 제때 반영하지 못하다 보니 팔면 팔수록 손해를 보는 '역마진' 구조가 지금의 에너지공기업의 부채를 키웠다는 비판도 나온다. 다만 정부는 여전히 물가 상황을 반영한 신중한 입장을 취하고 있다."

# STEP 4
# Encoding
inputs = tokenizer(body, return_tensors="pt", padding="max_length", truncation=True, max_length=1026)
# Generate Summary Text Ids
summary_text_ids = model.generate(
input_ids=inputs['input_ids'],
attention_mask=inputs['attention_mask'],
bos_token_id=model.config.bos_token_id,
eos_token_id=model.config.eos_token_id,
length_penalty=1.0,
max_length=100,
min_length=12,
num_beams=6,
repetition_penalty=1.5,
no_repeat_ngram_size=15,
)

# model_summary = summarization(body)
model_summary = tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
cls_title = classifier(title)
cls_summary = classifier(summary)
cls_model_summary = classifier(model_summary)

# STEP 5
print(model_summary)
# 분류 결과 중, 정확도가 제일 높은 항목으로 결과 표시 (기사 긍부정 분류)
print(cls_title)
print(cls_summary)
print(cls_model_summary)

# 기사 제목과의 유사도 분석 (제목 어그로 확인, 제목과 실제 내용이 유사한지, 요약 되어있는 summary와 모델로 요약한 model_summary와 비교)
# The sentences to encode
sentences = [
    title,
    summary,
    model_summary,
]

# 2. Calculate embeddings by calling model.encode()
embeddings = compare.encode(sentences)
# print(embeddings.shape)

# 3. Calculate the embedding similarities
similarities = compare.similarity(embeddings, embeddings)
print(similarities)

## 