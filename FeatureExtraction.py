# encoding:utf-8
'''
@author: kratos
@file: FeatureExtraction.py
@time: 2018/9/18 21:15
'''
# 提取训练集特征
import cv2
import os
import dlib
from skimage import io
import csv
import numpy as np
import pandas as pd

path_faces_rd = "data/faces/"
path_csv = "data/csvs/"
path_csv_feature_all = "data/features_all.csv"
PREDICTOR_PATH = 'dlib_dat/shape_predictor_68_face_landmarks.dat'
FACEREC_PATH = 'dlib_dat/dlib_face_recognition_resnet_model_v1.dat'


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(FACEREC_PATH)

def return_128d_features(path_img):
    img = io.imread(path_img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img_gray, 1)
    if len(dets) != 0:
        shape = predictor(img_gray, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img_gray, shape)
    else:
        face_descriptor = 0
    return face_descriptor

# 特征写入csv文件
def write_into_csv(path_faces_personX, path_csv):
    dir_pics = os.listdir(path_faces_personX)
    with open(path_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(dir_pics)):
            features_128d = return_128d_features(path_faces_personX + "/" + dir_pics[i])
            if features_128d == 0:
                i += 1
            else:
                writer.writerow(features_128d)

faces = os.listdir(path_faces_rd)
for person in faces:
    write_into_csv(path_faces_rd + person, path_csv + person + ".csv")

# 计算平均脸
def compute_the_mean(path_csv_rd):
    column_names = []
    for i in range(128):
        column_names.append("features_" + str(i + 1))
    rd = pd.read_csv(path_csv_rd, names=column_names)
    feature_mean = []
    for i in range(128):
        tmp_arr = rd["features_" + str(i + 1)]
        tmp_arr = np.array(tmp_arr)
        tmp_mean = np.mean(tmp_arr)
        feature_mean.append(tmp_mean)
    return feature_mean


with open(path_csv_feature_all, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    csv_rd = os.listdir(path_csv)
    print("特征均值: ")
    for i in range(len(csv_rd)):
        feature_mean = compute_the_mean(path_csv + csv_rd[i])
        # print(feature_mean)
        print(path_csv + csv_rd[i])
        writer.writerow(feature_mean)