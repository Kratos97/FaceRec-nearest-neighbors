# encoding:utf-8
'''
@author: kratos
@file: FaceRec.py
@time: 2018/9/20 19:33
'''
# 人脸识别
import dlib
import numpy as np
import cv2
import pandas as pd


path_csv_feature_all = "data/features_all.csv"
PREDICTOR_PATH = 'dlib_dat/shape_predictor_68_face_landmarks.dat'
FACEREC_PATH = 'dlib_dat/dlib_face_recognition_resnet_model_v1.dat'

facerec = dlib.face_recognition_model_v1(FACEREC_PATH)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cap = cv2.VideoCapture(0)
cap.set(3, 480)

# 计算欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    print(dist)
    if dist > 0.3:
        return "diff"
    else:
        return "same"

csv_rd = pd.read_csv(path_csv_feature_all, header=None)
features_known_arr = []

for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.ix[i, :])):
        features_someone_arr.append(csv_rd.ix[i, :][j])
    features_known_arr.append(features_someone_arr)

# 获取128维特征
def get_128d_features(img_gray):
    dets = detector(img_gray, 1)
    if len(dets) != 0:
        face_des = []
        for i in range(len(dets)):
            shape = predictor(img_gray, dets[i])
            face_des.append(facerec.compute_face_descriptor(img_gray, shape))
    else:
        face_des = []
    return face_des

while cap.isOpened():
    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    faces = detector(img_gray, 0)
    font = cv2.FONT_HERSHEY_COMPLEX
    pos_namelist = []
    name_namelist = []
    if len(faces) != 0:
        features_cap_arr = []
        for i in range(len(faces)):
            shape = predictor(img_rd, faces[i])
            features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))
        for k in range(len(faces)):
            name_namelist.append("unknown")
            pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
            for i in range(len(features_known_arr)):
                compare = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                if compare == "same":
                    name_namelist[k] = "kratos"
            for kk, d in enumerate(faces):
                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
        for i in range(len(faces)):
            cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    if kk == ord('q'):
        break
    cv2.imshow("camera", img_rd)
cap.release()
cv2.destroyAllWindows()