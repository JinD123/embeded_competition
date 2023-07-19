import time

import cv2
import numpy as np
from pipeline_module.core.base_module import BaseModule, TASK_DATA_CLOSE, TASK_DATA_OK, TaskData, TASK_DATA_SKIP, \
    TASK_DATA_IGNORE


class VideoModule(BaseModule):

    def __init__(self,socket, source=0, fps=25, skippable=False):
        super(VideoModule, self).__init__(skippable=skippable)
        self.task_stage = None
        self.source = source
        self.cap = None
        self.frame = None
        self.ret = False
        self.skip_timer = 0
        self.set_fps(fps)
        self.loop = True
        self.socket=socket
    def process_data(self, data):
        if not self.ret:
            if self.loop:
                self.open()
                return TASK_DATA_IGNORE
            else:
                return TASK_DATA_CLOSE
        data.source_fps = self.fps
        data.frame = self.frame
        # self.ret=True
        #
        # self.ret, self.frame = self.cap.read()
        client_socket, client_address = self.socket.accept()  # 接收一次消息
        # print('等待客户端连接...')
        #
        # print('客户端已连接:', client_address)

        # 接收图像数据流并解码
        image_data = b''
        while True:
            recv_data = client_socket.recv(1024)

            if not recv_data:
                break
            image_data += recv_data

            # 检查接收到的数据流是否已经包含完整的图片
            if image_data.endswith(b'\xff\xd9'):
                # 将接收到的数据流转换为图像
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        self.frame=image
        self.ret = True
        result = TASK_DATA_OK
        if self.skip_timer != 0:
            result = TASK_DATA_SKIP
            data.skipped = None
        skip_gap = int(self.fps * self.balancer.short_stab_interval)
        # skip_gap = 1
        # print(self.balancer.short_stab_module, self.balancer.short_stab_interval)
        if self.skip_timer > skip_gap:
            self.skip_timer = 0
        else:
            self.skip_timer += 1
        time.sleep(self.interval)
        return result

    def product_task_data(self):
        return TaskData(self.task_stage)

    def set_fps(self, fps):
        self.fps = 10
        # self.interval = 1 / 10
        self.interval = 0
    def open(self):
        super(VideoModule, self).open()
        # if self.cap is not None:
        #     self.cap.release()
        # self.cap = cv2.VideoCapture(self.source)
        # if self.cap.isOpened():
            # self.set_fps(self.cap.get(cv2.CAP_PROP_FPS))
            # self.ret, self.frame = self.cap.read()
        client_socket, client_address = self.socket.accept()  # 接收一次消息
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
            # self.ret=True

        self.frame = image
        self.ret=True
            # self.frame = cv2.imread("D:\\smart_classroom_demo-master\\silent_face\\images\\sample\\image_F1.jpg")
            # print("视频源帧率: ", self.fps)
        pass
