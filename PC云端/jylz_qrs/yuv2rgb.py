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
import numpy as np

# 读取 YUV 图像
yuv_img = cv2.imread('D:\\jylz_qrs\\UsePic_1920_1080_420.yuv', cv2.IMREAD_UNCHANGED)
height, width = yuv_img.shape[:2]

# 将 YUV 图像分离成 Y、U、V 三个通道
y_channel, u_channel, v_channel = cv2.split(yuv_img)

# 对 U、V 通道进行上采样
u_channel = cv2.resize(u_channel, (width, height), interpolation=cv2.INTER_CUBIC)
v_channel = cv2.resize(v_channel, (width, height), interpolation=cv2.INTER_CUBIC)

# 将 Y、U、V 三个通道合并成一个图像
yuv_img = cv2.merge([y_channel, u_channel, v_channel])

# 将 YUV 图像转换为 RGB 图像
rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB_I420)

# 显示 RGB 图像
cv2.imshow('RGB Image', rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()