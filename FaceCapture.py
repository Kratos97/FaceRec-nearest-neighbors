# encoding:utf-8
'''
@author: kratos
@file: FaceCapture.py
@time: 2018/9/16 20:23
'''
# 获取人脸图像并截取作为训练集
import dlib
import cv2
import numpy as np
import os
import shutil

PREDICTOR_PATH = 'dlib_dat/shape_predictor_68_face_landmarks.dat'


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
capture = cv2.VideoCapture(0)
capture.set(propId=3, value=480)


cnt_ss = 0
cnt_p = 0
current_face_dir = 0
path_make_dir = "data/faces/"
path_csv = "data/csvs/"
# 清空旧数据
def pre_clear():
    folders_rd = os.listdir(path_make_dir)
    for i in range(len(folders_rd)):
        shutil.rmtree(path_make_dir + folders_rd[i])
    csv_rd = os.listdir(path_csv)
    for i in range(len(csv_rd)):
        os.remove(path_csv + csv_rd[i])
pre_clear()

person_cnt = 0

while capture.isOpened():
    flag, im_rd = capture.read()
    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    font = cv2.FONT_HERSHEY_COMPLEX
    if kk == ord('n'):
        person_cnt += 1
        current_face_dir = path_make_dir + "kratos"
        print('\n')
        for dirs in (os.listdir(path_make_dir)):
            if current_face_dir == path_make_dir + dirs:
                shutil.rmtree(current_face_dir)
                print("删除旧的文件夹:", current_face_dir)
        os.makedirs(current_face_dir)
        print("新建的人脸文件夹: ", current_face_dir)
        cnt_p = 0

    if len(rects) != 0:
        for k, d in enumerate(rects):
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())
            hh = int(height / 2)
            ww = int(width / 2)
            cv2.rectangle(im_rd,
                          tuple([d.left() - ww, d.top() - hh]),
                          tuple([d.right() + ww, d.bottom() + hh]),
                          (0, 255, 0), 2)
            im_blank = np.zeros((height * 2, width * 2, 3), np.uint8)

            if kk == ord('s'):
                cnt_p += 1
                for ii in range(height * 2):
                    for jj in range(width * 2):
                        im_blank[ii][jj] = im_rd[d.top() - hh + ii][d.left() - ww + jj]
                cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_p) + ".jpg", im_blank)
                print("写入本地：", str(current_face_dir) + "/img_face_" + str(cnt_p) + ".jpg")

    cv2.putText(im_rd, "Faces: " + str(len(rects)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    if kk == ord('q'):
        break
    cv2.imshow("camera", im_rd)
capture.release()
cv2.destroyAllWindows()
