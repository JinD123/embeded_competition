import os
import time

import cv2
import numpy as np
import torch

try:
    from layers.functions.prior_box import PriorBox
    from models.faceboxes import FaceBoxes
    from utils.box_utils import decode
    from utils.nms.py_cpu_nms import py_cpu_nms as nms
except ImportError:
    from face_recog.layers.functions.prior_box import PriorBox
    from face_recog.models.faceboxes import FaceBoxes
    from face_recog.utils.box_utils import decode
    from face_recog.utils.nms.py_cpu_nms import py_cpu_nms as nms

cfg = {
    'name': 'FaceBoxes',
    # 'min_dim': 1024,
    # 'feature_maps': [[32, 32], [16, 16], [8, 8]],
    # 'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}
# 'name': 模型的名称，为'FaceBoxes'。
# 'min_sizes': 一个列表，包含三个子列表，分别表示每个特征图对应的基准框的尺寸。其中，第一个子列表表示第一个特征图对应的基准框尺寸，第二个子列表表示第二个特征图对应的基准框尺寸，以此类推。本例中，第一个特征图对应的基准框尺寸为32、64和128，第二个特征图对应的基准框尺寸为256，第三个特征图对应的基准框尺寸为512。
# 'steps': 一个列表，包含三个元素，分别表示每个特征图的步长。本例中，第一个特征图的步长为32，第二个特征图的步长为64，第三个特征图的步长为128。
# 'variance': 一个列表，包含两个元素，分别表示回归目标的方差。本例中，第一个元素为0.1，第二个元素为0.2。
# 'clip': 一个布尔值，表示是否对预测框的坐标进行裁剪。本例中，'clip'为False，表示不进行裁剪。
# 'loc_weight': 一个浮点数，表示回归损失函数的权重系数。本例中，'loc_weight'的值为2.0。
# 'gpu_train': 一个布尔值，表示是否使用GPU进行训练。本例中，'gpu_train'为True，表示使用GPU进行训练。


class FaceBoxesLocation:
    def __init__(self, weights='weights/FaceBoxes.pth', **opt):
        # ** opt表示将opt字典中的所有键值对作为关键字参数传递给函数。
        # 具体来说，opt可以包含以下一些可选字段：
        # 'device': 字符串类型，表示模型所在的设备，可以是'cpu'
        # 'cuda'。默认值为'cuda'。
        # 'confidence_threshold': 浮点数类型，表示目标检测的置信度阈值，当预测框的置信度小于该阈值时，会将该预测框丢弃。默认值为0.05。
        # 'nms_threshold': 浮点数类型，表示非极大值抑制的阈值，当两个预测框的重叠度（即IoU）大于该阈值时，会将置信度较低的预测框丢弃。默认值为0.3。
        # 'top_k': 整数类型，表示模型输出的预测框的数量，当预测框的数量超过该值时，会进行Top - k筛选。默认值为5000。
        # 这些参数可以用来调整目标检测模型的性能和行为，以适应不同的应用场景和任务要求。
        # 如果没有传入opt参数，则会将空字典{}
        # 作为默认值传递给 ** opt，不会报错。在代码中，可以通过使用opt.get(key, default_value)
        # 的方式获取opt字典中的参数，如果opt字典中不存在相应的键，则会返回默认值default_value。因此，即使没有传入opt参数，代码中使用opt.get(key, default_value)
        # 的方式获取参数仍然是安全的。例如，在__init__函数中，使用opt.get('device', 'cuda')
        # 的方式获取设备参数，如果没有传入opt参数，则会使用默认值 'cuda'。
        net = FaceBoxes(phase='test', size=None, num_classes=2)  # 首先创建分类检测网络
        net = load_model(net, opt.get("weights", weights), False)#加载网络权重参数,FalseB表示加载到GPU上
        net.eval()
        # net.eval()
        # 是PyTorch中一个用于模型评估的方法。它的作用是将模型设置为评估模式（evaluationmode），即关闭模型中的训练特性（如Dropout、BatchNormalization等）。
        # 具体来说，当模型设置为评估模式时，PyTorch会做以下几件事情：
        # 将所有Dropout和Batch
        # Normalization层设置为“固定”，即使用训练时的参数进行计算，不再进行随机采样或批标准化。
        # 将requires_grad属性设置为False，即关闭梯度计算，避免在评估模式下误用反向传播算法。
        # 不再在每个batch后清空梯度，因为评估模式下不需要进行梯度更新。
        # 总之，net.eval()的作用是确保模型在评估时的行为与训练时的行为一致，并且不会出现不必要的计算或梯度更新，以保证评估结果的准确性。
        self.net = net
        #self.top_k用于控制目标检测模型输出的预测框的数量
        self.top_k = opt.get('top_k', 5000)
        #一些参数配置
        # self.confidence_threshold: 目标检测的置信度阈值，当预测框的置信度小于该阈值时，会将该预测框丢弃。
        #
        # self.nms_threshold: 非极大值抑制的阈值，当两个预测框的重叠度（即IoU）大于该阈值时，会将置信度较低的预测框丢弃。
        #
        # self.keep_top_k: 模型输出的预测框的数量，当预测框的数量超过该值时，会进行Top - k筛选，只保留置信度最高的keep_top_k个预测框。
        self.confidence_threshold = opt.get('confidence_threshold', 0.05)
        self.nms_threshold = opt.get('nms_threshold', 0.3)
        self.keep_top_k = opt.get('keep_top_k', 750)

    def face_location(self, img, resize=1, cof=0.5):
        # 处理图片
        img = np.float32(img)
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        # 前向传播
        #net就是网络模型，输入参数就会前向传播
        loc, conf = self.net(img)  # forward pass
        #
        priors = priorbox.forward()
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        # 该代码从loc中获取位置偏移量，prior_data中获取先验框信息，以及cfg['variance']
        # 中获取方差信息，然后使用这些信息对网络预测的边界框进行解码，得到实际的目标边界框。
        # 其中，loc是网络输出的位置偏移量，是一个大小为[n, 4]
        # 的张量，其中n表示待检测的目标数量；prior_data是先验框信息，是一个大小为[num_priors, 4]
        # 的张量，其中num_priors表示先验框的数量；cfg['variance']
        # 是方差信息，是一个长度为4的列表，用于对位置偏移量进行还原。
        # 具体来说，代码首先通过squeeze函数将loc的第一维度（即batch_size）压缩掉，然后调用decode函数进行解码。
        # decode函数会将位置偏移量loc和先验框信息prior_data作为输入，然后根据网络预测的位置偏移量和先验框的大小和位置，
        # 计算出实际的目标边界框。
        #
        # 具体的解码过程：
        # 将先验框的中心点坐标和宽高信息转换为左上角和右下角的坐标；
        #
        # 使用方差信息对位置偏移量进行还原，得到实际的偏移量；
        #
        # 将实际的偏移量应用到先验框上，得到实际的偏移后的边界框坐标；
        #
        # 将实际的边界框坐标转换为中心点坐标和宽高信息，并将其转换为左上角和右下角的坐标。
        # 最终，decode函数会返回解码后的目标边界框，即boxes。
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        scores = scores[order]
        # do NMS
        #将检测框的坐标信息和置信度信息合并成一个数组
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, self.nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        # 筛选出置信度较高的人脸
        dets = dets[dets[:, 4] > cof, :4]
        return dets#返回所有检测人脸的位置


def check_keys(model, pretrained_state_dict):
    #这段代码的主要作用是检查预训练模型是否与当前模型匹配，以避免出现权重加载错误或缺失的情况。如果检查通过，则返回True，表示可以成功加载预训练模型。
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


pretrained_path = '../weights/FaceBoxes.pth'
face_source = '../people'

if __name__ == '__main__':
    fbl = FaceBoxesLocation(weights=pretrained_path)
    imgs = os.listdir(face_source)
    imgs = [os.path.join(face_source, img) for img in imgs]
    for frame in (cv2.imread(fn) for fn in imgs):
        start_time = time.time()
        img_shape = frame.shape
        resize_w = 640
        resize_rate = resize_w / img_shape[1]
        resize_h = int(img_shape[0] * resize_rate)#将宽度设置为640时应有的高度
        resize_frame = cv2.resize(frame, (resize_w, resize_h))
        # 人脸定位
        face_locations = fbl.face_location(frame)#得到非极大值抑制后并且置信度较高的结果
        face_locations[:, :4] *= resize_rate

        for x1, y1, x2, y2 in face_locations:
            # Draw a box around the face
            cv2.rectangle(resize_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            # font = cv2.FONT_HERSHEY_DUPLEX
        end_time = time.time()
        cv2.imshow("show", resize_frame)
        print(f'用时: {round(end_time - start_time, 2)} s')
        while cv2.waitKey(-1) and 0xFF == 'p':
            pass
