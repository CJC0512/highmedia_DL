from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis


face = FaceAnalysis()
face.prepare(ctx_id=0, det_size=(640,640))  # 자동 크기 조절

target_face = []

app = FastAPI()
@app.post("/registFace/")
async def registFace(file: UploadFile):
    content = await file.read()
    # STEP 3
    # img = cv2.imread("iu1.jpg")
    # --> buf = file.open("iu1.jpg")
    # --> img = cv2.imdecode(buf)
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
    # STEP 4
    faces1 = face.get(img)
    assert len(faces1)==1
    # STEP 5
    target_face.append(np.array(faces1[0].normed_embedding, dtype=np.float32))
    print(target_face)
    return {"result":len(faces1)}
    
    
@app.post("/compareFace/")
async def compareFace(file: UploadFile):
    content = await file.read()
    # STEP 3
    # img = cv2.imread("iu1.jpg")
    # --> buf = file.open("iu1.jpg")
    # --> img = cv2.imdecode(buf)
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
    # STEP 4
    faces1 = face.get(img)
    assert len(faces1)==1
    # STEP 5
    test_face = np.array(faces1[0].normed_embedding, dtype=np.float32)
    sim = np.dot(target_face[0], test_face.T)
    return {"result":sim.item()}