""" ━━━━━━神兽出没━━━━━━ 
 　　　┏┓　　　┏┓ 
 　　┃　　　　　　　┃ 
 　　┃　　　━　　　┃ 
 　　┃　┳┛　┗┳　┃ 
 　　┃　　　　　　　┃ 
 　　┃　　　┻　　　┃ 
 　　┃　　　　　　　┃ 
 　　┗━┓　　　┏━┛Code is far away from bug with the animal rotecting 
 　　　　┃　　　┃ 神兽保佑,代码无bug 
 　　　　┃　　　┃ 
 　　　　┃　　　┗━━━┓ 
 　　　　┃　　　　　　　┣┓ 
 　　　　┃　　　　　　　┏┛ 
 　　  　┗┓┓┏━┳┓┏┛ 
　　　　　┃┫┫　┃┫┫ 
　　　　　┗┻┛　┗┻┛ 
"""
import cv2 as cv
import numpy as np

#创建记录读取的训练数据集的列表
images = []

#往列表里添加训练集图片,这里的地址根据数据集在你电脑上放置的位置进行修改，数目可以增加，数目越多识别准确度越高。
images.append(cv.imread("data\\1.3.jpg", cv.IMREAD_GRAYSCALE))
images.append(cv.imread("data\\1.jpg", cv.IMREAD_GRAYSCALE))
images.append(cv.imread("data\\1.4.jpg", cv.IMREAD_GRAYSCALE))
images.append(cv.imread("data\\1.5.jpg", cv.IMREAD_GRAYSCALE))
images.append(cv.imread("data\\1.6.jpg", cv.IMREAD_GRAYSCALE))

images.append(cv.imread("data\\2.jpg", cv.IMREAD_GRAYSCALE))
images.append(cv.imread("data\\3.jpg", cv.IMREAD_GRAYSCALE))
#创建标签
labels = [0,0,0,0,0,1,1]
images[0] = np.resize(images[0], (200, 200))
images[1] = np.resize(images[1], (200, 200))
images[2] = np.resize(images[2], (200, 200))
images[3] = np.resize(images[3], (200, 200))
images[4] = np.resize(images[4], (200, 200))
images[5] = np.resize(images[5], (200, 200))
images[6] = np.resize(images[6], (200, 200))
#利用opencv中的FisherFace算法，生成Fisher识别器模型
recognizer1 = cv.face_EigenFaceRecognizer.create()
#利用opencv中的EigenFace算法，生成Eign识别器模型


#训练数据集
recognizer1.train(images,np.array(labels))


#读取待检测的图像
predict_image1 = cv.imread("6.jpg",cv.IMREAD_GRAYSCALE)
predict_image1 = np.resize(predict_image1, (200, 200))

#识别图像
label1,configence1 = recognizer1.predict(predict_image1)


#输出识别结果
print("label1= ",label1)
print("confidence1= ",configence1)
