import cv2
import os
import numpy as np
# import mysql.connector
# 加载数据集
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(int(filename.split(".")[0]))
    return images, labels
# 训练模型
def train_model(images, labels, num_components=40000):
    # 将图像转换为向量
    # print(np.size(images,1))


    images[0]=np.resize(images[0],(200,200))
    images[1] = np.resize(images[0],(200, 200))
    # images[2] = np.resize(images[0], (200, 200))
    # images[3] = np.resize(images[0], (200, 200))
    print(np.size(images[0]))
    print(np.size(images[1]))
    data = np.array(images).reshape((-1, images[0].shape[0]*images[0].shape[1])).astype(np.float32)
    # 计算平均脸


    mean, eigenvectors = cv2.PCACompute(data, mean=None, maxComponents=num_components)
    # 计算特征脸
    eigenfaces = eigenvectors.reshape((-1, images[0].shape[0], images[0].shape[1])).astype(np.uint8)
    # 投影数据
    data_projected = cv2.PCAProject(data, mean, eigenvectors)
    # 存储模型
    model = cv2.face.EigenFaceRecognizer_create(num_components=num_components)
    model.train(data_projected, np.array(labels))
    model.write("eigenface_model.yml")
    return mean, eigenfaces, model
# 进行人脸识别
def recognize_face(image, mean, eigenfaces, model):
    # 将图像转换为向量
    image = np.resize(image, (200, 200))
    data = np.array(image).reshape((1, image.shape[0]*image.shape[1])).astype(np.float32)


    print(np.size(data))
    # 投影数据
    eigenfaces=np.float32(eigenfaces)
    mean=np.float32(mean)

    data_projected = cv2.PCAProject(data, mean, eigenfaces)
    # 进行识别
    label, confidence = model.predict(data_projected)
    return label, confidence
# 将结果存储到MySQL数据库中
# def save_to_database(label, confidence):
#     # 连接MySQL数据库
#     cnx = mysql.connector.connect(user='username', password='password',
#                                   host='127.0.0.1', database='database_name')
#     cursor = cnx.cursor()
#     # 插入数据
#     query = "INSERT INTO recognition_results (label, confidence) VALUES (%s, %s)"
#     cursor.execute(query, (label, confidence))
#     cnx.commit()
#     # 关闭连接
#     cursor.close()
#     cnx.close()
# 加载数据集
images, labels = load_images_from_folder("data")


# 训练模型
mean, eigenfaces, model = train_model(images, labels)
# 进行人脸识别并将结果存储到数据库中
num_components = eigenfaces.shape[0]
eigenfaces = np.reshape(eigenfaces, (num_components, -1))
image = cv2.imread("2.jpg", cv2.IMREAD_GRAYSCALE)

label, confidence = recognize_face(image, mean, eigenfaces, model)
print(label)
print(confidence)
# save_to_database(label, confidence)