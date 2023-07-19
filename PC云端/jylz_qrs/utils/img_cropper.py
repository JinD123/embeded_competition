import numpy as np
from cv2 import cv2


class CropImage:
    @staticmethod
    def get_new_box(src_w, src_h, bbox, scale, out_w, out_h):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]
        center_x, center_y = box_w / 2 + x, box_h / 2 + y
        aspect_src = box_w / box_h
        aspect_target = out_w / out_h
        # 调整边框比例
        if aspect_src > aspect_target:
            box_h = box_w / aspect_target
        else:
            box_w = box_h * aspect_target

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2
        # 调整边框位置
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1

        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        return int(left_top_x), int(left_top_y), \
               int(right_bottom_x), int(right_bottom_y)

    @staticmethod
    def crop(org_img, bbox, scale, out_w, out_h, crop=True, return_box=False):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
            left_top_x, left_top_y, right_bottom_x, right_bottom_y = bbox
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
            right_bottom_x, right_bottom_y = CropImage.get_new_box(src_w, src_h, bbox, scale, out_w, out_h)

            img = org_img[left_top_y: right_bottom_y + 1,
                  left_top_x: right_bottom_x + 1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return (dst_img, [left_top_x, left_top_y, right_bottom_x, right_bottom_y]) if return_box else dst_img
    #crop 的值为 True，则需要进行裁剪操作。首先，根据原始图像的宽度 src_w、高度 src_h、待裁剪区域的边框 bbox、裁剪比例 scale、目标宽度 out_w 和目标高度 out_h，
    # 通过一个名为 get_new_box 的函数计算出裁剪后的新的边框信息 left_top_x、left_top_y、right_bottom_x、right_bottom_y，然后根据新的边框信息将原始图像中
    # 的相应区域裁剪出来，得到裁剪后的图像 img，然后对图像进行缩放操作，得到目标大小的图像 dst_img。最后，将裁剪后的新的边框信息 left_top_x、left_top_y、
    # right_bottom_x、right_bottom_y 作为输出的边框信息（如果需要返回的话），否则只返回目标大小的图像 dst_img。