import math

import numpy as  np
import torch

default_focus = torch.tensor([1176.03538718, 165.21773522])


# ==================
# 基于关键点的伸手识别部分
# ==================
class CheatingActionAnalysis:
    @staticmethod
    def stretch_out_degree(keypoints, left=True, right=True, focus=None):
        """
        :param focus: 透视焦点
        :param keypoints: Halpe 26 keypoints 或 136关键点 [N,keypoints]
        :param left: 是否计算作弊的伸手情况
        :param right: 是否计算右臂的伸手情况
        :return: ([N,left_hand_degree],[N,left_hand_degree]), hand_degree = (arm out?,forearm out?,straight arm?)
        """
        if focus is None:
            shoulder_vec = keypoints[:, 6] - keypoints[:, 5]
        else:
            shoulder_vec = (keypoints[:, 6] + keypoints[:, 5]) / 2 - focus
        result = []
        if left:
            arm_vec = keypoints[:, 5] - keypoints[:, 7]
            forearm_vec = keypoints[:, 7] - keypoints[:, 9]
            _results = torch.hstack([torch.cosine_similarity(shoulder_vec, arm_vec).unsqueeze(1),
                                     torch.cosine_similarity(shoulder_vec, forearm_vec).unsqueeze(1),
                                     torch.cosine_similarity(arm_vec, forearm_vec).unsqueeze(1)])
            result.append(_results)
        if right:
            shoulder_vec = -shoulder_vec
            arm_vec = keypoints[:, 6] - keypoints[:, 8]
            forearm_vec = keypoints[:, 8] - keypoints[:, 10]
            _results = torch.hstack([torch.cosine_similarity(shoulder_vec, arm_vec).unsqueeze(1),
                                     torch.cosine_similarity(shoulder_vec, forearm_vec).unsqueeze(1),
                                     torch.cosine_similarity(arm_vec, forearm_vec).unsqueeze(1)])
            result.append(_results)
        return result

    @staticmethod
    def is_stretch_out(degree, threshold=None, dim=1):
        """
        判定单手是否伸手
        :param degree: 通过stretch_out_degree计算的手臂余弦相似度数组
        :param threshold: 阈值
        :param dim: 处理的维度
        :return:
        """
        if threshold is None:
            threshold = torch.tensor([0.8, 0.8, 0.5])
        return torch.all(degree > threshold, dim=dim)

    @staticmethod
    def is_passing(keypoints):
        """
        是否在左右传递物品
        :param keypoints: Halpe 26 keypoints 或 136关键点 [N,keypoints]
        :return: [N,左右传递?]+1 左传递，-1 右边传递 0 否
        """
        irh = CheatingActionAnalysis.is_raise_hand(keypoints)
        stretch_out_degree_L, stretch_out_degree_R = CheatingActionAnalysis.stretch_out_degree(keypoints)
        isoL = CheatingActionAnalysis.is_stretch_out(stretch_out_degree_L)
        isoR = CheatingActionAnalysis.is_stretch_out(stretch_out_degree_R)

        left_pass = isoL & ~irh[:, 0]  # 是否是左传递
        left_pass_value = torch.zeros_like(left_pass, dtype=int)
        left_pass_value[left_pass] = 1
        right_pass = isoR & ~irh[:, 1]  # 是否是右传递
        right_pass_value = torch.zeros_like(right_pass, dtype=int)
        right_pass_value[right_pass] = -1
        return left_pass_value + right_pass_value

    @staticmethod
    def is_raise_hand(keypoints):
        """
        腕部高过头顶时为举手，因为坐标远点是左上角，所以是头部y坐标大于腕部的时候为举手
        :param keypoints: keypoints: Halpe 26 keypoints 或 136关键点 [N,keypoints]
        :return: [N,[左手举手?,右手举手?]]
        """
        return keypoints[:, [17], 1] > keypoints[:, [9, 10], 1]


# ==================
# 基于关键点的转头识别部分
# ==================

# f1 = np.array([[0, 1]])

depth_correction_factor = 0.28550474951641663


# depth_correction_factor = 0.3


def turn_head_angle(rvec, tvec):
    # print(head_pose[0][0], head_pose[0][1], head_pose[0][2])
    # f2 = torch.tensor([[head_pose[1][0][0], ]])
    # print(torch.arccos(torch.cosine_similarity(f1, f2)))
    # print(head_pose[1])
    x = tvec[0][0]
    depth = tvec[2][0] * depth_correction_factor
    if depth < 0:
        depth, x = -depth, -x
    right_front_angle = math.acos(depth / math.sqrt((x * x + depth * depth)))
    # print(x, depth, right_front_angle, rvec[1][0])
    if x < 0:
        return rvec[1][0] + right_front_angle
    else:
        return rvec[1][0] - right_front_angle


def turn_head_angle_yaw(yaw, tvec):
    # print(head_pose[0][0], head_pose[0][1], head_pose[0][2])
    # f2 = torch.tensor([[head_pose[1][0][0], ]])
    # print(torch.arccos(torch.cosine_similarity(f1, f2)))
    # print(head_pose[1])
    x = tvec[0][0]
    depth = tvec[2][0] * depth_correction_factor
    if depth < 0:
        depth, x = -depth, -x
    right_front_angle = math.acos(depth / math.sqrt((x * x + depth * depth))) * 57.3
    print(x, depth, right_front_angle, yaw)
    if x < 0:
        return yaw - right_front_angle
    else:
        return yaw + right_front_angle


if __name__ == '__main__':
    p1, p2, p3, p4 = (74, 182), (271, 179), (386, 359), (757, 268)
    A = np.array([[p2[1] - p1[1], p1[0] - p2[0]],
                  [p4[1] - p3[1], p3[0] - p4[0]]])
    b = np.array([(p1[0] - p2[0]) * p1[1] - (p1[1] - p2[1]) * p1[0],
                  (p3[0] - p4[0]) * p3[1] - (p3[1] - p4[1]) * p3[0]])
    print(np.linalg.solve(A, b))
