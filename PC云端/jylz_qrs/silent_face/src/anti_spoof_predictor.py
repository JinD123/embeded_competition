import os

import torch
import torch.nn.functional as F

try:
    from src.data_io import transform as trans
    from src.utility import get_kernel, parse_model_name
    from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
except ImportError:
    from silent_face.src.data_io import transform as trans
    from silent_face.src.utility import get_kernel, parse_model_name
    from silent_face.src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


class AntiSpoofPredictor:
    def __init__(self, device_id, model_path):
        super(AntiSpoofPredictor, self).__init__()
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() and device_id in ['0', '1', '2', '3'] else "cpu")
        self._load_model(model_path)

    def _load_model(self, model_path):
        # define model
        #2.7_80x80_MiniFASNetV2
        model_name = os.path.basename(model_path)#os.path.basename函数获取传入的model_path参数的文件名，将其赋值给model_name变量
        h_input, w_input, model_type, _ = parse_model_name(model_name)#parse_model_name函数解析model_name，返回一个四元组，其中前
        # 两个元素为h_input和w_input，分别表示输入图片的高和宽，第三个元素model_type表示模型类型。
        # 由于返回的四元组中的最后一个元素用不到，故使用占位符_表示不需要使用的变量。
        #从文件名上解析h_input=80,w_input=80 model_type=MiniFASNetV2,_=2.7
        self.kernel_size = get_kernel(h_input, w_input, )
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img):
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result
