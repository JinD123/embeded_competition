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
import argparse

import os
import platform
import sys
import numpy as np
from pathlib import Path
FILE = Path(__file__).resolve()# __file__指的是当前文件(即detect.py),FILE最终保存着当前文件的绝对路径,比如D://yolov5/detect.py
ROOT = FILE.parents[0]   #  ROOT保存着当前项目的父目录,比如 D://yolov5
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # ROOT设置为相对路径

from models.common import DetectMultiBackend
from yolov5_utils.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5_utils.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5_utils.utils.plots import Annotator, colors, save_one_box
from yolov5_utils.utils.torch_utils import select_device, smart_inference_mode
import torch
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

import cv2
import education
from PyQt5.QtCore import QDate, QTime, Qt ,QDateTime
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QTimer, QTime

class Education(QMainWindow):
    def __init__(self):

        super().__init__()
        #yolo参数
        self.hide_labels = None
        self.hide_conf = None
        self.vid_writer = None
        self.vid_path = None
        self.view_img = None
        self.save_img = None
        self.save_crop = None
        self.save_conf = None
        self.save_txt = None
        self.line_thickness = None
        self.device = None
        self.vid_stride = None
        self.dataset = None
        self.source = None

        self.iou_thres = None
        self.classes = None
        self.augment = None
        self.agnostic_nms = None
        self.save_dir = None
        self.conf_thres = None
        self.max_det = None
        self.visualize = None
        self.project = None
        self.path="D:\\jylz_qrs\\data\\result\\result.jpg"
        self.pt = None
        self.imgsz = None
        self.stride = None
        self.names = None
        self.model = None
        #是否在一直打卡监听状态
        self.pc_is_signing=None
        self.time = QTime.currentTime()#获取当前时间
        self.timerdate = QTimer()#创建计时器
        self.timerdate.timeout.connect(self.showtime)  # 连接计时器timeout信号到显示时间函数
        self.timerdate.start(1000)#启动定时器
        self.ui = education.Ui_Education()
        self.ui.setupUi(self)
        self.ui.PCsignwindow.hide()
        self.ui.PCclasswindow.hide()
        self.ui.PCexamwindow.hide()
        self.ui.hisconsignwindow.hide()
        self.ui.hisconclasswindow.hide()
        self.result=None
        self.ui.hisconexamwindow.hide()
        #实现各个menu选项action对应的界面弹出
        self.ui.action_sign_pc.triggered.connect(self.sign_pc_function)
        self.ui.action_exam_pc.triggered.connect(self.exam_pc_function)
        self.ui.action_class_pc.triggered.connect(self.class_pc_function)
        self.ui.action_exam_hiscon.triggered.connect(self.exam_hiscon_function)
        self.ui.action_sign_hiscon.triggered.connect(self.sign_hiscon_function)
        self.ui.action_class_hiscon.triggered.connect(self.class_hiscon_function)
        #创建摄像头
        self.cap = None
        self.image = None
        #创建yolo模型参数
        self.opt = self.parse_opt("D:\\jylz_qrs\\data\\result")
        

    #界面切换功能
    def sign_pc_function(self):
        self.ui.PCsignwindow.show()
        self.ui.PCclasswindow.hide()
        self.ui.PCexamwindow.hide()
        self.ui.hisconsignwindow.hide()
        self.ui.hisconclasswindow.hide()
        self.ui.hisconexamwindow.hide()

        self.set_model(**vars(self.opt))
        self.pc_is_signing=True
        #构建模型


        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            exit()
        while self.pc_is_signing:
            # 读取一帧
            ret, frame = self.cap.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.
            self.image=rgb_frame

            cv2.imwrite("D:\\jylz_qrs\\data\\result\\result.jpg", frame)
            # 如果帧读取失败则退出循环
            if not ret:
                break
            # 将帧转换为QImage格式
            #进行检测处理
            self.detect_face()
            rgb_frame=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            image = QImage(rgb_frame, rgb_frame.shape[1], rgb_frame.shape[0], self.image.strides[0], QImage.Format_RGB888)
            # 将QImage格式转换为QPixmap格式
            pixmap = QPixmap.fromImage(image)
            # 在QLabel控件上显示图像
            self.ui.pc_picture_sign.setPixmap(pixmap)
            # 使QApplication进入事件循环
            QApplication.processEvents()
            # 释放摄像头并关闭窗口
        self.cap.release()
        cv2.destroyAllWindows()
    def exam_pc_function(self):
        self.ui.PCsignwindow.hide()
        self.ui.PCclasswindow.hide()
        self.ui.PCexamwindow.show()
        self.ui.hisconsignwindow.hide()
        self.ui.hisconclasswindow.hide()
        self.ui.hisconexamwindow.hide()
        self.pc_is_signing = False

    def class_pc_function(self):
        self.ui.PCsignwindow.hide()
        self.ui.PCclasswindow.show()
        self.ui.PCexamwindow.hide()
        self.ui.hisconsignwindow.hide()
        self.ui.hisconclasswindow.hide()
        self.ui.hisconexamwindow.hide()
        self.pc_is_signing = False
    def exam_hiscon_function(self):
        self.ui.PCsignwindow.hide()
        self.ui.PCclasswindow.hide()
        self.ui.PCexamwindow.hide()
        self.ui.hisconsignwindow.hide()
        self.ui.hisconclasswindow.hide()
        self.ui.hisconexamwindow.show()
        self.pc_is_signing = False
    def sign_hiscon_function(self):
        self.ui.PCsignwindow.hide()
        self.ui.PCclasswindow.hide()
        self.ui.PCexamwindow.hide()
        self.ui.hisconsignwindow.show()
        self.ui.hisconclasswindow.hide()
        self.ui.hisconexamwindow.hide()
        self.pc_is_signing = False
    def class_hiscon_function(self):
        self.ui.PCsignwindow.hide()
        self.ui.PCclasswindow.hide()
        self.ui.PCexamwindow.hide()
        self.ui.hisconsignwindow.hide()
        self.ui.hisconclasswindow.show()
        self.ui.hisconexamwindow.hide()
        self.pc_is_signing = False

    #实时显示时间功能
    def showtime(self):
        self.time = QDateTime.currentDateTime()
        timestr = self.time.toString("yyyy-MM-dd HH:mm:ss")  # 设置时间格式
        self.ui.lcd_time.display(timestr)  # 显示时间


    #构建yolo模型
    def set_model(self,
                  weights=ROOT / 'face.pt',  # model path or triton URL
                  source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
                  data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                  imgsz=(640, 640),  # inference size (height, width)
                  conf_thres=0.25,  # confidence threshold
                  iou_thres=0.45,  # NMS IOU threshold
                  max_det=1000,  # maximum detections per image
                  device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                  view_img=False,  # show results
                  save_txt=False,  # save results to *.txt
                  save_conf=False,  # save confidences in --save-txt labels
                  save_crop=False,  # save cropped prediction boxes
                  nosave=False,  # do not save images/videos
                  classes=None,  # filter by class: --class 0, or --class 0 2 3
                  agnostic_nms=False,  # class-agnostic NMS
                  augment=False,  # augmented inference
                  visualize=False,  # visualize features
                  update=False,  # update all models
                  project=ROOT / 'runs/detect',  # save results to project/name
                  name='exp',  # save results to project/name
                  exist_ok=False,  # existing project/name ok, do not increment
                  line_thickness=3,  # bounding box thickness (pixels)
                  hide_labels=False,  # hide labels
                  hide_conf=False,  # hide confidences
                  half=False,  # use FP16 half-precision inference
                  dnn=False,  # use OpenCV DNN for ONNX inference
                  vid_stride=1,  # video frame-rate stride
    ):
        device = select_device('0')
        self.device = device
        self.save_txt=save_txt
        self.hide_conf=hide_conf
        self.hide_labels=hide_labels
        self.model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        self.save_crop=save_crop
        self.save_conf=save_conf

        self.augment = augment
        self.stride= self.model.stride
        self.names=self.model.names
        self.pt = self.model.pt

        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        self.project = project
        self.vid_stride =vid_stride
        self.visualize = visualize
        self.max_det = max_det
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.source = source
        self.save_img = not nosave and not self.source.endswith('.txt')
        self.line_thickness=line_thickness
        self.view_img = view_img
        bs=1
        self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt, vid_stride=self.vid_stride)
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs
        self.save_dir = increment_path(Path(self.project) / name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else 1, 3, *self.imgsz))  # warmup




        #进行人脸检测
    def detect_face(self):
        #对模型进行预热，对模型的一些初始化，以提高一些速度，将一个随机张量输入给模型，使模型的权重与缓存与推理时相同

        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        #注意模型的输入是经过裁剪、缩放、归一化等预处理后的
        for path, im, im0s, vid_cap, s in self.dataset:

            with dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:

                self.visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False

                pred = self.model(im, augment=self.augment, visualize=self.visualize)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Second
            #
            # -stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                p, im0, frame = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # im.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            print(s)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                       
                    if self.save_crop:
                        save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
          # Save results (image with detections)
            if self.save_img:
                if self.dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
            self.image = im0

            # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / 1 * 1E3 for x in dt)  # speeds per image

        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

    def parse_opt(self,source):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'face.pt', help='model path or triton URL')
        parser.add_argument('--source', type=str, default=source, help='file/dir/URL/glob/screen/0(webcam)')
        parser.add_argument('--data', type=str, default=ROOT / 'data/face.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt






#创建yolo参数



    # def pc_clicked(self):
