import cv2
import dlib
import imutils
import face_recognition
import os
import time
import numpy as np
import math
from imutils import face_utils


#计算两个坐标的距离
def euclidian_distance(p1, p2):
    diff_x = abs(p2[0]-p1[0])
    diff_y = abs(p2[1]-p1[1])
    return math.sqrt(diff_x*diff_x + diff_y*diff_y)


# 使用 Dlib 的正面人脸检测器 frontal_face_detector
detector = dlib.get_frontal_face_detector()
#脸部关键点预测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

path=r"xs.mp4"
cap = cv2.VideoCapture(path)
# cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS) # not supported by my webcam
print(fps)
count = int(cap.get(7))  # 总帧数
print("总帧数",count)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(width,height)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # codec
# cv2.VideoWriter( filename, fourcc, fps, frameSize )
out = cv2.VideoWriter('outputxs.mp4', fourcc, fps, (width, height))

#(mouth_lower, mouth_upper) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']
(mouth_lower, mouth_upper) = (48,60)

# ACTIVATION_RATIO = 3.0
ACTIVATION_RATIO = 3

color = (0, 0, 255)
actived=False
actived1=False
actived2=False
actived3=False
result=[]
result0=[]
result1=[]
result2=[]
result3=[]
sum=0
sum1=0
sum2=0
sum3=0
j=0
j1=0
j2=0
j3=0
fras=0

guo_image = face_recognition.load_image_file("guo.png")
guo_face_encoding = face_recognition.face_encodings(guo_image)[0]

yu_image = face_recognition.load_image_file("yu.png")
yu_face_encoding = face_recognition.face_encodings(yu_image)[0]


known_face_encodings = [guo_face_encoding,yu_face_encoding]
known_face_names = ["guo","yu"]
#初始化
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
maintalk=[]
mainactive=[]
notalk=False
while(True):

    if (fras >=count):
        break

    ret, frame = cap.read()
    # 使用 detector 检测器来检测图像中的人脸
    faces = detector(frame, 1)
    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # rgb_small_frame = small_frame[:, :, ::-1]#转为RGB
    # face_locations = face_recognition.face_locations(rgb_small_frame)
    # print(face_locations)
    # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)


    if(len(faces)>0):
        if fras >50:
            notalk=False
            mainactive=[]
            maintalk=[]
            maxindex=None
            actived,actived1,actived2,actived3=False,False,False,False
            if len(result0)>50 and (j+50)<len(result0):
                for i in range(j, j + 50):
                    sum = sum + result0[i]
                if sum > 25:
                    maintalk.append(sum)
                    mainactive.append("guo")
                sum = 0
                j = j + 1
            if len(result1)>50 and (j1+50)<len(result1):
                for i in range(j1, j1 + 50):
                    sum1 = sum1 + result1[i]
                if sum1 > 25:
                    maintalk.append(sum1)
                    mainactive.append("yu")
                sum1 = 0
                j1 = j1 + 1
            if(len(maintalk)>0):
                maxindex = maintalk.index(max(maintalk))
                print(maintalk)
                print(mainactive)
                print("索引是", maxindex)
                if mainactive[maxindex] == "guo":
                    actived = True
                elif mainactive[maxindex] == "yu":
                    actived1 = True
            else:
                notalk=True



        for (i, face) in enumerate(faces):
            # color = (0, 0, 255)
            # print("人脸位置：",face,"第%d个"%(i))
            # print("人脸数 / faces in all：", len(faces))
            # print(face) [(161, 76) (546, 461)]
            # fx, fy, fw, fh=face
            fx=face.left()
            fy=face.top()

            # print(face.top(),face.right(),face.bottom(),face.left())
            left=face.left()
            right=face.right()
            bottom=face.bottom()
            top=face.top()
            faceloc=(top,right,bottom,left)

            sb_face_encoding = face_recognition.face_encodings(frame, [faceloc])[0]
            # cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)


            # 确定面部区域的面部地标，然后将面部标志（x，y）坐标转换成NumPy阵列
            # 使用predictor来计算面部轮廓
            shape = predictor(frame, face)
            shape = face_utils.shape_to_np(shape)


            mleft = (shape[48, 0], shape[48, 1])#49和55
            mright = (shape[54, 0], shape[54, 1])

            mtop = (shape[51, 0], shape[51, 1])#58和52
            mbottom = (shape[57, 0], shape[57, 1])

            diff_h = euclidian_distance(mtop, mbottom)
            diff_w = euclidian_distance(mleft, mright)

            ratio = round(diff_w / diff_h, 5)

            # cv2.putText(frame, "ratio: %s" % ratio, (0, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            # 绘制嘴巴的点
            # for (x, y) in shape[mouth_lower:mouth_upper]:
            #     cv2.circle(frame, (x, y), 1, color, -3)
            matches = face_recognition.compare_faces(known_face_encodings, sb_face_encoding, tolerance=0.6)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, sb_face_encoding)
            best_match_index = np.argmin(face_distances)  # 最小值的下标
            # print("最小距离下标为%d" % (best_match_index))

            if matches[best_match_index] and (face_distances[best_match_index]<=0.5):
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0),1)

            if actived and name=="guo":
                cv2.putText(frame, "talking", (fx+20, fy-30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            elif actived1 and name=="yu":
                cv2.putText(frame, "talking", (fx + 20, fy-30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            if notalk:
                # cv2.putText(frame, "no talking", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                print("1")



            if ratio >= ACTIVATION_RATIO :  # mouth shut & not talking
                # print("闭着嘴巴")
                if name=="guo":
                    result0.append(0)
                    # cv2.putText(frame, "shut", (fx + 20, fy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif name=="yu":
                    result1.append(0)
                    # cv2.putText(frame, "shut", (fx + 20, fy + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # print("张开嘴巴")
                if name=="guo":
                    result0.append(1)
                elif name=="yu":
                    result1.append(1)
    else:
        cv2.putText(frame, "no face", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    fras=fras+1
    print("当前帧",fras)
    #显示画面
    # cv2.imshow('frame',frame)
    #写入视频
    out.write(frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

