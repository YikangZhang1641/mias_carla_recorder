import numpy as np
import cv2
from matplotlib import pyplot as plt

imgname1 = '/home/mias/Datasets/carla_pothole2/2023_04_20 18:07:03/rgb_front_left/485_765805691.jpeg'
imgname2 = '/home/mias/Datasets/carla_pothole2/2023_04_20 18:07:03/rgb_front_right/485_765805691.jpeg'
#实例化的sift函数
sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread(imgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) #灰度处理图像
img2 = cv2.imread(imgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)#灰度处理图像
hmerge = np.hstack((gray1, gray2)) #水平拼接
cv2.imshow("gray", hmerge) #拼接显示为gray
cv2.waitKey(0)

kp1, des1 = sift.detectAndCompute(img1,None)   #des是描述子
#sift.detectAndComputer(gray， None)计算出图像的关键点和sift特征向量   参数说明：gray表示输入的图片
#des1表示sift特征向量，128维
print("图片1的关键点数目："+str(len(kp1)))
#print(des1.shape)
kp2, des2 = sift.detectAndCompute(img2,None)  #des是描述子
print("图片2的关键点数目："+str(len(kp2)))

img3 = cv2.drawKeypoints(img1,kp1,img1,color=(255,0,255)) #画出特征点，并显示为红色圆圈
#img3 = cv2.drawKeypoints(gray, kp, img) 在图中画出关键点   参数说明：gray表示输入图片, kp表示关键点，img表示输出的图片
#print(img3.size)
img4 = cv2.drawKeypoints(img2,kp2,img2,color=(255,0,255)) #画出特征点，并显示为红色圆圈

hmerge = np.hstack((img3, img4)) #水平拼接
cv2.imshow("point", hmerge) #拼接显示为gray
cv2.waitKey(0)

# BFMatcher解决匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
#print(matches)
#print(len(matches))
# 调整ratio
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
#img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
img5 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imshow("BFmatch", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()
