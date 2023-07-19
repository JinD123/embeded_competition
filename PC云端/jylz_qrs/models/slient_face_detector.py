import os

import numpy as np

from silent_face.src.anti_spoof_predictor import AntiSpoofPredictor
from silent_face.src.utility import parse_model_name
from utils.img_cropper import CropImage

wanted_model_index = [0 ,1]


class SilentFaceDetector:
    def __init__(self, device_id='gpu', model_dir='weights/anti_spoof_models'):
        self.models = []#使用到的模型列表
        self.params = []#使用到的模型
        for i, model_name in enumerate(os.listdir(model_dir)):#i为文件编号，model_name为文件名称
            if i not in wanted_model_index:
                continue
            self.models.append(AntiSpoofPredictor(device_id, os.path.join(model_dir, model_name)))
            #用相应权重文件将对应模型创立并加入到models列表中，
            self.params.append(parse_model_name(model_name))#解析出模型相关参数并加入到参数列表

    def detect(self, frame, face_location):
        face_location = face_location.copy()
        face_location[2:] = face_location[2:] - face_location[:2]
        prediction = np.zeros((1, 3))
        # sum the prediction from single model's result
        for model, (h_input, w_input, model_type, scale) in zip(self.models, self.params):
            param = {
                "org_img": frame,
                "bbox": face_location,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = CropImage.crop(**param)
            prediction += model.predict(img)

        # draw result of prediction
        label = np.argmax(prediction)
        return label, prediction[0][label] / len(self.models)
