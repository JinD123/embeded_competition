import json
import os
import socket as socket1
from threading import Thread

import cv2
import numpy as np
from PIL import ImageDraw, Image, ImageFont
from PyQt5 import QtCore
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QInputDialog, QLineEdit, QMessageBox

from face_recog.models import face_recog
from face_recog.models.face_boxes_location import FaceBoxesLocation
from models.pose_estimator import PnPPoseEstimator
from models.slient_face_detector import SilentFaceDetector
from smart_classroom.list_items import FaceListItem
from ui.face_register import Ui_FaceRegister
from utils.common import validFileName, read_mask_img, read_encoding_json2npy, read_img_from_cn_path

face_bank_base_dir = 'resource\\face_bank'


class Face_RegisterApp(QWidget, Ui_FaceRegister):
    fontText = ImageFont.truetype("resource/font/NotoSansCJKkr-Black.otf",
                                  20,
                                  encoding="utf-8")
    fontTextBig = ImageFont.truetype("resource/font/NotoSansCJKkr-Black.otf",
                                     50,
                                     encoding="utf-8")
    add_new_name_signal = QtCore.pyqtSignal(object, object)
    show_frame_signal = QtCore.pyqtSignal(object)
    increase_process_signal = QtCore.pyqtSignal()
    decrease_process_signal = QtCore.pyqtSignal()
    init_process_signal = QtCore.pyqtSignal()
    send_data_signal = QtCore.pyqtSignal()
    def __init__(self, socket,socket2,parent=None):
        super(Face_RegisterApp, self).__init__(parent)
        self.setupUi(self)
        self.server_socket=socket
        self.cap = None
        self.thread = None
        self.is_adding_new_name = False
        #负责发送信息
        self.server_socket2 = socket2

        self.image=None
        # 读取人型边框
        self.human_boarder_mask = read_mask_img('resource/pic/human_boarder.png')
        # 初始化人脸数据库
        self.init_face_bank()
        # 信号槽
        self.add_new_name_signal.connect(self.add_new_name)
        self.show_frame_signal.connect(self.show_frame)
        # self.send_data_signal.connect(self.start_send_data_thread)
        # 进度条处理
        self.init_process_signal.connect(
            lambda: self.register_completeness_pb.setValue(self.register_completeness_pb.minimum()))
        self.decrease_process_signal.connect(lambda: self.register_completeness_pb.setValue(
            max(self.register_completeness_pb.minimum(),
                self.register_completeness_pb.value() - 1)))
        self.increase_process_signal.connect(lambda: self.register_completeness_pb.setValue(
            min(self.register_completeness_pb.maximum(),
                self.register_completeness_pb.value() + 1)))

        # 删除学生按钮
        self.delete_student_btn.clicked.connect(lambda: self.delete_face())

        # 过滤
        def student_list_filter(txt: str, self=self):
            txt = txt.strip()
            if txt == '':
                for i in range(self.student_list.count()):
                    self.student_list.item(i).setHidden(False)
            else:
                for i in range(self.student_list.count()):
                    item = self.student_list.item(i)
                    item.setHidden(item.name.find(txt) < 0)

        self.student_list_filter_txt.textChanged.connect(student_list_filter)

        self.class_list_filter_txt.textChanged.connect(lambda: self.refresh_face_bank())

    def refresh_face_bank(self):
        """
        刷新人脸库
        """
        try:
            self.face_bank_list_cbx.clear()
            txt = self.class_list_filter_txt.text()
            for bank_name in self.face_banks:
                if txt == '' or bank_name.find(txt) > -1:
                    self.face_bank_list_cbx.addItem(bank_name)
        except Exception as e:
            print('refresh_face_bank:', e)

    def init_face_bank(self):
        """
        初始化人脸库
        """
        #

        try:

            self.face_banks = os.listdir(face_bank_base_dir)
            self.refresh_face_bank()
            self.face_bank_list_cbx.currentTextChanged.connect(self.open_face_bank)
            # 初始化
            self.refresh_face_bank_btn.clicked.connect(lambda: self.open_face_bank())
            self.refresh_face_bank_btn.setIcon(QIcon(':/func/refresh.ico'))
            self.init_process()
            print("初始化成功")
        except Exception as e:
            print('init_face_bank:', e)

    def delete_face(self):
        try:
            for item in self.student_list.selectedItems():
                item.delete_name()
            self.open_face_bank()
        except Exception as e:
            print(e)

    def open_face_bank(self, bank_name=None):
        try:
            self.student_list.clear()
            bank_name = bank_name if bank_name is not None else self.face_bank_list_cbx.currentText()
            face_bank_dir = os.path.join(face_bank_base_dir, bank_name)
            self.known_face_names, self.known_faces_data = self.get_known_faces_data(face_bank_dir)
            for idx, (name, encoding, face_img) in enumerate(self.known_faces_data):
                FaceListItem(self.student_list,
                             face_img,
                             encoding,
                             name,
                             idx,
                             face_bank_dir,
                             self.known_face_names
                             ).add_item()
        except Exception as e:
            print('open_face_bank: ', e)

    @staticmethod
    def get_known_faces_data(facebank):
        try:
            known_face_names = os.listdir(facebank)  # 读取已经录入的人名
            print(known_face_names)
            known_faces_data = [(name, read_encoding_json2npy(
                os.path.join(facebank, name,'encoding.json')
            ), read_img_from_cn_path(
                os.path.join(facebank, name,'face.jpg')
            )) for name in known_face_names]  # 人名，编码，图片

            return known_face_names, known_faces_data
        except Exception as e:
            print('get_known_faces_data', e)
    # def send_data2_function(self):
    #     data2 = "123"
    #     client_socket2, client_address2 = self.server_socket2.accept()
    #     print(f"Received connection from {client_address2[0]}:{client_address2[1]}")
    #     client_socket2.send(data2.encode())
    #     print("发送")

    # def start_send_data_thread(self):
    #     send_data_thread = QThread()
    #     send_data_thread.started.connect(self.send_data_function)
    #     send_data_thread.start()
    def add_new_name(self, img, encoding):
        try:
            #注意每个人是在face_bank下的一个库文件夹下
            face_bank_dir = os.path.join(face_bank_base_dir, self.face_bank_list_cbx.currentText())

            new_name, okPressed = QInputDialog.getText(self, "人像采集完成", "请输入名称:", QLineEdit.Normal, "")
            new_name = new_name.strip()
            if okPressed and self.check_input_name(new_name):
                # 重命名文件
                path = os.path.join(face_bank_dir, new_name)
                os.mkdir(path)
                # 保存面部编码
                with open(os.path.join(path, 'encoding.json'), 'w') as f:
                    json.dump(encoding, f)
                # 保存面部图片
                cv2.imencode('.jpg', img)[1].tofile(os.path.join(path, 'face.jpg'))
                self.open_face_bank()
            elif okPressed:
                QMessageBox.information(self,
                                        "未注册成功",
                                        "不要输入已经存在的名字或者不输入，名称只能包含中文、字母和数字",
                                        QMessageBox.Yes | QMessageBox.No)
        except Exception as e:
            print('add_new_name:', e)
        finally:
            self.is_adding_new_name = False

    def check_input_name(self, name):
        return validFileName(name) and name not in self.known_face_names

    def face_register_process(self):
        """
        人脸注册过程
        """

        print("打开摄像头")
        fbl = FaceBoxesLocation()
        pnp = PnPPoseEstimator()
        sfd = SilentFaceDetector()
        # 开启摄像头
        # cap = cv2.VideoCapture(0)
        # self.cap = cap
    # if cap.isOpened():
        # ret, frame = cap.read()

        frame = cv2.imread("1.jpg")
        if frame.all() != None:
            ret = True

        while ret and self.running:
            try:
                # 摄像头反转

                frame = cv2.flip(frame, 1)
                orig_frame = frame.copy()
                # 人脸检测
                face_locations = fbl.face_location(frame).astype(int)
                #face_location是人脸所在位置
                if face_locations.shape[0] > 1 and self.biggest_face_ckb.isChecked():
                    box_width = face_locations[:, 2] - face_locations[:, 0]
                    face_locations = face_locations[box_width == np.max(box_width)]
                # 静默活体识别
                silent_face_detections = [sfd.detect(orig_frame, face_location) for face_location in face_locations]

                if self.is_adding_new_name:
                    tips_text = '正在注册'
                    self.init_process()
                elif len(face_locations) == 0:
                    tips_text = '无人在摄像头前'
                    self.init_process()
                elif len(face_locations) > 1:
                    tips_text = '请保证镜头前只有一个人'
                    self.init_process()
                else:

                    tips_text = ''
                    face_landmarks = face_recog.face_landmarks(frame, face_locations)
                    face_keypoints = np.array(face_landmarks, dtype=np.float)
                    # 边界范围
                    x1, y1, x2, y2 = face_locations[0]
                    # 头部姿态
                    r_vec, t_vec = pnp.solve_pose(face_keypoints[0])
                    # 绘制头部姿态
                    pnp.draw_axis(frame, r_vec, t_vec)
                    euler = pnp.get_euler(r_vec, t_vec)
                    tolerance = 50
                    if self.is_register_ckb.isChecked():
                        if silent_face_detections[0][0] != 1:
                            tips_text = '假脸无法进行注册'
                            self.decrease_process()
                        elif abs(320 - (x1 + x2) // 2) > tolerance or abs(200 - (y1 + y2) // 2) > tolerance or abs(
                                175 - (x2 - x1)) > tolerance or abs(230 - (y2 - y1)) > tolerance:
                            tips_text = '请位于人形框内'
                            self.decrease_process()
                        elif abs(euler[0][0]) > 20 or abs(euler[1][0]) > 20 or (abs(euler[2][0]) > 20 and abs (euler[2][0]) < 170):
                            tips_text = "请正视摄像头"
                            self.decrease_process()
                        else:
                            self.increase_process()
                            if self.process_completeness() >= 0.999:
                                encoding = face_recog.face_encodings(frame, face_locations)[0]
                                self.is_adding_new_name = True
                                self.add_new_name_signal.emit(orig_frame[y1:y2, x1:x2], encoding.tolist())
                                self.init_process()
                                continue
                # 视频显示
                # opencv不支持中文，这里使用PIL作为画板

                frame_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(frame_pil)  # 创建画板
                for (x1, y1, x2, y2), (label, prob) in zip(face_locations, silent_face_detections):
                    # 静默活体检测结果
                    if label == 1:
                        result_text = "真脸 置信度: {:.2f}".format(prob)
                        color = (255, 0, 0)
                    else:
                        result_text = "假脸 置信度: {:.2f}".format(prob)
                        color = (0, 0, 255)
                    fontText = ImageFont.truetype("resource/font/NotoSansCJKkr-Black.otf",
                                                  int((x2 - x1) * 0.2),
                                                  encoding="utf-8")
                    # 把人脸框出来标号
                    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                    _, f_h = fontText.getsize(result_text)
                    draw.text((x1, y1 - f_h), result_text, color, fontText)
                # 绘制文字提示
                if self.center_tips_text_ckb.isChecked():
                    f_w, f_h = self.fontTextBig.getsize(tips_text)
                    draw.text((int((640 - f_w) / 2), int((480 - f_h) / 2)), tips_text, (0, 0, 255), self.fontTextBig)
                else:
                    draw.text((0, 0), tips_text, (0, 0, 255), self.fontText)
                # 显示图片
                frame_show = np.array(frame_pil)
                # 添加人形边框
                if self.human_boarder_ckb.isChecked():
                    frame_show[self.human_boarder_mask] = np.array([255, 255, 255] if tips_text == '' else [0, 0, 255])
                self.show_frame_signal.emit(frame_show)
                print("显示人脸")
            except Exception as e:
                print(e)
            finally:
                # 下一帧
                # ret, frame = cap.read()
                print("来接受")
                client_socket, client_address = self.server_socket.accept()  # 接收一次消息

                # print('等待客户端连接...')
                #
                # print('客户端已连接:', client_address)
                print("接收中")
                # 接收图像数据流并解码
                image_data = b''
                while True:
                    data = client_socket.recv(1024)

                    if not data:

                        break
                    image_data += data


                    # 检查接收到的数据流是否已经包含完整的图片


                    print(data)
                    if data.endswith(b'\xff\xd9'):
                        # 将接收到的数据流转换为图像
                        nparr = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        break

                image=cv2.resize(image,(640,480))

                frame = image
                #向下位机发送信息
                # self.send_data_signal.emit()
                print("接收下一帧")
                # cv2.imshow("im",image)
                # while True:
                #     key = cv2.waitKey(1) & 0xFF
                #
                #     # 如果按下 'q' 键，退出循环
                #     if key == ord('q'):
                #         break

                # 关闭窗口
                # cv2.destroyWindow('im')
                # data = b'is registering'
                # if (client_socket == 0):
                #     self.server_socket.send(data)
                # ret = True


    def init_process(self):
        self.init_process_signal.emit()

    def process_completeness(self):
        max_value = self.register_completeness_pb.maximum()
        min_value = self.register_completeness_pb.minimum()
        cur_value = self.register_completeness_pb.value()
        return (cur_value - min_value) / (max_value - min_value)

    def increase_process(self):
        self.increase_process_signal.emit()

    def decrease_process(self):
        self.decrease_process_signal.emit()

    def show_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.video_screen.width() - 10, self.video_screen.height() - 10))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        self.video_screen.setPixmap(QPixmap.fromImage(frame))

        # 每个线程对应有对应的线程函数，即其一直执行的任务
    def close(self) -> bool:  # 定义close方法，返回值为bool类型
        if self.thread is not None:  # 如果对应对象线程非空，证明该对象的该线程需要关闭
            self.running = False  # 置标志位为false
            self.thread.join()  # 阻塞当前线程，等待线程结束
            self.thread = None  # 置为空
        if self.cap is not None:
            self.cap.release()  # 将打开的摄像头资源释放
            self.cap = None  # 摄像头置为空
        super(Face_RegisterApp, self).close()#调用Qwidget类的close方法，关闭界面

    def open(self):
        self.open_face_bank()
        self.running = True
        # 启动
        frame = np.zeros((480, 640, 3), np.uint8)
        (f_w, f_h), _ = cv2.getTextSize("Opening", cv2.FONT_HERSHEY_TRIPLEX, 1, 2)
        # 字符串 字体格式 默认大小 线条宽度，返回值f_w,f_h为返回的宽度与高度
        cv2.putText(frame, "Opening", (int((640 - f_w) / 2), int((480 - f_h) / 2)),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将要放上去的图片重新resize一下大小以便可以在QLabel类型的video_screen上较美观的显示
        frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        self.video_screen.setPixmap(QPixmap.fromImage(frame))

        # 启动线程，对应的线程函数为face_register_process
        self.thread = Thread(target=self.face_register_process, args=())
        self.thread.start()
