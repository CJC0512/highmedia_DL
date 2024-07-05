# STEP 1 : 모듈 가져오기
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# STEP 2 : 모델 가져오기 (자동 다운로드)
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640,640))  # 자동 크기 조절

# STEP 3 
# from insightface.data import get_image as ins_get_image
# img = ins_get_image('t1')
img1 = cv2.imread("winter1.jpg")
img2 = cv2.imread("winter2.jpg")

# STEP 4 추론
# faces = app.get(img1)    # 5개의 Task를 돌림.
# assert len(faces)==6

faces1 = app.get(img1)    # 5개의 Task를 돌림.
assert len(faces1)==1

faces2 = app.get(img2)    # 5개의 Task를 돌림.
assert len(faces2)==1

print(faces1[0])

# STEP 5
# rimg = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", rimg)

rimg = app.draw_on(img1, faces1)
cv2.imwrite("./t1_output.jpg", rimg)


# then print all-to-all face similarity
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)
# feats = np.array(feats, dtype=np.float32)
# sims = np.dot(feats, feats.T)
# print(sims)

# embedding 된 데이터 유사도 분석 코드 (-1 ~ 1) => 어느정도 유사한지 알려주는 것(퍼센트X 단순거리O)
feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces2[0].normed_embedding, dtype=np.float32)
sims = np.dot(feat1, feat2.T)
print(sims)
