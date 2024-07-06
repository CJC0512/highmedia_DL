from fastapi import FastAPI, Form
# STEP 1
from transformers import pipeline

# STEP 2
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

app = FastAPI()


@app.post("/textClassification/")
async def login(text: str = Form()):
    
    # STEP 3
    # text = "현대바이오, '폴리탁셀' 코로나19 치료 가능성에 19% 급등" # positive

    # STEP 4
    result = classifier(text)

    # STEP 5
    print(result)
    
    return {
        "text" : text,
        "result": result
        }