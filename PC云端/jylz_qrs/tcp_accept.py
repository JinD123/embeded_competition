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
# import cv2
# import numpy as np
# import socket
#
# # 创建socket
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
# # 绑定socket到一个可用的端口上
# server_address = ('192.168.200.1', 8888)  # 设置PC端的IP地址和端口号
# server_socket.bind(server_address)
#
# # 监听连接
# server_socket.listen(1)
#
# # 接受客户端连接
# print('等待客户端连接...')
# client_socket, client_address = server_socket.accept()
# print('客户端已连接:', client_address)
#
# # 接收图像数据流并解码
# image_data = b''
# while True:
#     data = client_socket.recv(1024)
#     if not data:
#         break
#     image_data += data
#
# # 将接收到的数据流转换为图像
# nparr = np.frombuffer(image_data, np.uint8)
# image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
# # 保存解码后的图像为JPEG文件
# cv2.imwrite('解码后的图像.jpg', image)
#
# # 显示解码后的图像
# cv2.imshow('解码后的图像', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 关闭socket
# client_socket.close()
# server_socket.close()
import cv2
import numpy as np
import socket

# 创建socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定socket到一个可用的端口上
server_address = ('192.168.200.1', 8888)  # 设置PC端的IP地址和端口号
server_socket.bind(server_address)

# 监听连接
server_socket.listen(1)

while True:
    # 接受客户端连接
    client_socket, client_address = server_socket.accept()  # 接收一次消息
    # print('等待客户端连接...')
    #
    # print('客户端已连接:', client_address)

    # 接收图像数据流并解码
    image_data = b''
    while True:
        data = client_socket.recv(1024)


        if not data:

            break
        image_data += data

        # 检查接收到的数据流是否已经包含完整的图片
        if image_data.endswith(b'\xff\xd9'):
            # 将接收到的数据流转换为图像
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 保存解码后的图像为JPEG文件
            cv2.imwrite('1.jpg', image)

            # 显示解码后的图像
            # cv2.imshow("1",image)
            # cv2.waitKey(0)

            # 清空图像数据，准备接收下一张图片
            image_data = b''


    # 关闭客户端socket
client_socket.close()

# 关闭服务器socket
server_socket.close()
