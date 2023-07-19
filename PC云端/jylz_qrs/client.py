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
import cv2
import socket

SERVER_IP = '192.168.200.1'
SERVER_PORT = 8888

def send_image():
    # 读取图像文件
    image = cv2.imread('result.jpg')

    # 编码图像为JPEG格式的字节流
    _, encoded_image = cv2.imencode('.jpg', image)

    # 获取字节流数据
    image_data = encoded_image.tobytes()

    # 创建TCP客户端套接字
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 连接到服务器
    client_socket.connect((SERVER_IP, SERVER_PORT))

    # 发送图像数据
    client_socket.sendall(image_data)

    print("Image data sent successfully.")

    # 关闭套接字
    client_socket.close()

if __name__ == '__main__':
    send_image()