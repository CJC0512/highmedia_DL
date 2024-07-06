# STEP 1
from transformers import pipeline

# STEP 2
# question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model") # stevhliu 가 자기 이름 까먹음
question_answerer = pipeline("question-answering", model="timpal0l/mdeberta-v3-base-squad2") 

# STEP 3
# question = "How many programming languages does BLOOM support?"
# context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

question = "가는 말이 고와야 오는 말이 곱다 는 무슨 의미지?"
context = "‘내가 남에게 잘해야 남도 나에게 잘한다’는 뜻을 가진 ‘가는 말이 고와야 오는 말이 곱다’라는 우리 속담을 ‘가는 말이 고우면 얕본다’는 명언을 제시하며 먼저 친절을 베풀었음에도 불구하고 얕잡아보는 이들을 일침 했다."


# STEP 4
result = question_answerer(question=question, context=context)

# STEP 5
print(result)