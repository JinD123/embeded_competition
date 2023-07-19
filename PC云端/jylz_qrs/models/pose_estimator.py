import cv2
import numpy as np
import torch

from models.face_aligner import PFLDFaceAligner, MobileNetSEFaceAligner
from utils.alphapose.transforms import heatmap_to_coord_simple_regress
from utils.img_cropper import CropImage
from utils.simple_transform import SimpleTransform


class AlphaPoseEstimator:
    empty_tensor = torch.tensor([])

    def __init__(self, weights, device, input_size=(256, 192), output_size=(64, 48), face_aligner_weights=None):
        self._input_size = input_size
        self._output_size = output_size
        self._sigma = 2
        self.device = device
        self.model = torch.jit.load(weights).to(device)
        self.transformation = SimpleTransform(
            scale_factor=0,
            input_size=self._input_size,
            output_size=self._output_size,
            rot=0, sigma=self._sigma, add_dpg=False, gpu_device=self.device)
        self.heatmap_to_coord = heatmap_to_coord_simple_regress
        # 预热
        _ = self.model(torch.zeros(1, 3, 256, 192).to(device))
        self.face_aligner = MobileNetSEFaceAligner(face_aligner_weights,
                                                   device) if face_aligner_weights is not None else None

    def preprocess(self, frame, detections):
        inps = []
        cropped_boxes = []
        for i in range(detections.shape[0]):
            box = detections[i, :4]
            inp, cropped_box = self.transformation.test_transform(frame, box)
            inps.append(torch.FloatTensor(inp))
            cropped_boxes.append(torch.FloatTensor(cropped_box))
        return torch.stack(inps, dim=0).to(self.device), torch.stack(cropped_boxes, dim=0).to(self.device)

    def estimate(self, frame, detections):
        if detections.shape[0] <= 0:
            return self.empty_tensor, self.empty_tensor
        inps, cropped_boxes = self.preprocess(frame, detections)
        # 关键点检测
        hm = self.model(inps).cpu().detach()

        self.eval_joints = [*range(0, 136)]
        pose_coords = []
        pose_scores = []
        norm_type = 'sigmoid'
        hm_size = self._output_size
        for i in range(hm.shape[0]):
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = self.heatmap_to_coord(hm[i][self.eval_joints], bbox, hm_shape=hm_size,
                                                           norm_type=norm_type)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        preds_kps = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)
        if self.face_aligner is not None:
            face_bboxes = self.get_face_boxes(preds_kps)
            face_bboxes[:, 2:] = face_bboxes[:, 2:] - face_bboxes[:, :2]

            face_cropped_and_bboxes = [CropImage.crop(frame, bbox, 1.1, 56, 56, return_box=True) for bbox in
                                       face_bboxes]
            face_bboxes = torch.tensor([bbox for _, bbox in face_cropped_and_bboxes], dtype=torch.float32)
            face_bboxes[:, 2:] = face_bboxes[:, 2:] - face_bboxes[:, :2]
            face_cropped = np.array(
                [cropped_img.tolist() for cropped_img, _ in face_cropped_and_bboxes],
                dtype=np.uint8)[:, :, :, ::-1]
            face_landmarks = self.face_aligner.align(face_cropped)
            face_landmarks *= face_bboxes[:, 2:].unsqueeze(1)
            face_landmarks += face_bboxes[:, :2].unsqueeze(1)
            preds_kps[:, 26:94] = face_landmarks[:, :]
            pass
        return preds_kps, preds_scores

    @staticmethod
    def get_face_boxes(keypoints):
        face_keypoints = keypoints[:, 26:94]
        face_outline_keypoints = face_keypoints[:, :27]
        x_min = torch.min(face_outline_keypoints[:, :, 0], dim=1).values
        y_min = torch.min(face_outline_keypoints[:, :, 1], dim=1).values
        x_max = torch.max(face_outline_keypoints[:, :, 0], dim=1).values
        y_max = torch.max(face_outline_keypoints[:, :, 1], dim=1).values
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    @staticmethod
    def get_face_keypoints(face_keypoints):
        """
        获取标准化后的人脸关键点坐标
        :param face_keypoints: 脸部关键点
        :return: 标准化后的人脸关键点坐标，人脸框的位置
        """
        face_outline_keypoints = face_keypoints[:27]
        face_x1 = torch.min(face_outline_keypoints[:, 0])
        face_y1 = torch.min(face_outline_keypoints[:, 1])
        face_x2 = torch.max(face_outline_keypoints[:, 0])
        face_y2 = torch.max(face_outline_keypoints[:, 1])
        # 获取标准化的脸部坐标
        face_x1_y1 = torch.tensor([face_x1, face_y1])
        face_width = torch.tensor([face_x2 - face_x1, face_y2 - face_y1])
        scaled_face_keypoints = (face_keypoints - face_x1_y1) / face_width
        return scaled_face_keypoints, (face_x1, face_y1, face_x2, face_y2)


class PnPPoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        # 3D model points.
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Mouth left corner
            (150.0, -150.0, -125.0)  # Mouth right corner
        ]) / 4.5

        self.model_points_68 = self._get_full_model_points()

        self.body_model_points = np.array([  # 18,19,5,6
            (0.0, -40.0, 0.0),  # neck
            (0.0, 240.0, 5),  # hip
            (-80, 0.0, 0.5),  # left shoulder
            (+80, 0.0, 0.5),  # right shoulder
        ])

        self.neck_model_points = np.array([  # 17,18,5,6
            (0.0, -90.0, 5),  # head
            (0.0, 30.0, 0.0),  # neck
            (-60, 70.0, 5),  # left shoulder
            (+60, 70.0, 5),  # right shoulder
        ])

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])
        # self.r_vec = None
        # self.t_vec = None

    @staticmethod
    def _get_full_model_points():
        """Get all 68 3D model points from file"""
        # raw_value = []
        # with open(filename) as file:
        #     for line in file:
        #         raw_value.append(line)
        # model_points = np.array(raw_value, dtype=np.float32)
        # model_points = np.reshape(model_points, (3, -1)).T
        #
        # # Transform the model into a front view.
        # model_points[:, 2] *= -1
        model_points = np.array([
            [-73.393523, -29.801432, -47.667532],
            [-72.775014, -10.949766, -45.909403],
            [-70.533638, 7.929818, -44.84258],
            [-66.850058, 26.07428, -43.141114],
            [-59.790187, 42.56439, -38.635298],
            [-48.368973, 56.48108, -30.750622],
            [-34.121101, 67.246992, -18.456453],
            [-17.875411, 75.056892, -3.609035],
            [0.098749, 77.061286, 0.881698],
            [17.477031, 74.758448, -5.181201],
            [32.648966, 66.929021, -19.176563],
            [46.372358, 56.311389, -30.77057],
            [57.34348, 42.419126, -37.628629],
            [64.388482, 25.45588, -40.886309],
            [68.212038, 6.990805, -42.281449],
            [70.486405, -11.666193, -44.142567],
            [71.375822, -30.365191, -47.140426],
            [-61.119406, -49.361602, -14.254422],
            [-51.287588, -58.769795, -7.268147],
            [-37.8048, -61.996155, -0.442051],
            [-24.022754, -61.033399, 6.606501],
            [-11.635713, -56.686759, 11.967398],
            [12.056636, -57.391033, 12.051204],
            [25.106256, -61.902186, 7.315098],
            [38.338588, -62.777713, 1.022953],
            [51.191007, -59.302347, -5.349435],
            [60.053851, -50.190255, -11.615746],
            [0.65394, -42.19379, 13.380835],
            [0.804809, -30.993721, 21.150853],
            [0.992204, -19.944596, 29.284036],
            [1.226783, -8.414541, 36.94806],
            [-14.772472, 2.598255, 20.132003],
            [-7.180239, 4.751589, 23.536684],
            [0.55592, 6.5629, 25.944448],
            [8.272499, 4.661005, 23.695741],
            [15.214351, 2.643046, 20.858157],
            [-46.04729, -37.471411, -7.037989],
            [-37.674688, -42.73051, -3.021217],
            [-27.883856, -42.711517, -1.353629],
            [-19.648268, -36.754742, 0.111088],
            [-28.272965, -35.134493, 0.147273],
            [-38.082418, -34.919043, -1.476612],
            [19.265868, -37.032306, 0.665746],
            [27.894191, -43.342445, -0.24766],
            [37.437529, -43.110822, -1.696435],
            [45.170805, -38.086515, -4.894163],
            [38.196454, -35.532024, -0.282961],
            [28.764989, -35.484289, 1.172675],
            [-28.916267, 28.612716, 2.24031],
            [-17.533194, 22.172187, 15.934335],
            [-6.68459, 19.029051, 22.611355],
            [0.381001, 20.721118, 23.748437],
            [8.375443, 19.03546, 22.721995],
            [18.876618, 22.394109, 15.610679],
            [28.794412, 28.079924, 3.217393],
            [19.057574, 36.298248, 14.987997],
            [8.956375, 39.634575, 22.554245],
            [0.381549, 40.395647, 23.591626],
            [-7.428895, 39.836405, 22.406106],
            [-18.160634, 36.677899, 15.121907],
            [-24.37749, 28.677771, 4.785684],
            [-6.897633, 25.475976, 20.893742],
            [0.340663, 26.014269, 22.220479],
            [8.444722, 25.326198, 21.02552],
            [24.474473, 28.323008, 5.712776],
            [8.449166, 30.596216, 20.671489],
            [0.205322, 31.408738, 21.90367],
            [-7.198266, 30.844876, 20.328022]])

        return model_points

    def show_3d_model(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = Axes3D(fig)
        model_points = self.model_points_68
        x = model_points[:, 0]
        y = model_points[:, 1]
        z = model_points[:, 2]

        ax.scatter(x, y, z)
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        pyplot.show()

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        assert image_points.shape[0] == self.model_points_68.shape[
            0], "3D points and 2D points should be of same number."
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)

        # (success, rotation_vector, translation_vector) = cv2.solvePnP(
        #     self.model_points,
        #     image_points,
        #     self.camera_matrix,
        #     self.dist_coeefs,
        #     rvec=self.r_vec,
        #     tvec=self.t_vec,
        #     useExtrinsicGuess=True)
        return rotation_vector, translation_vector

    def solve_body_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        assert image_points.shape[0] == self.body_model_points.shape[
            0], "3D points and 2D points should be of same number."
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.body_model_points, image_points, self.camera_matrix, self.dist_coeefs)

        # (success, rotation_vector, translation_vector) = cv2.solvePnP(
        #     self.model_points,
        #     image_points,
        #     self.camera_matrix,
        #     self.dist_coeefs,
        #     rvec=self.r_vec,
        #     tvec=self.t_vec,
        #     useExtrinsicGuess=True)
        return rotation_vector, translation_vector

    def solve_neck_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        assert image_points.shape[0] == self.neck_model_points.shape[
            0], "3D points and 2D points should be of same number."
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.body_model_points, image_points, self.camera_matrix, self.dist_coeefs)

        # (success, rotation_vector, translation_vector) = cv2.solvePnP(
        #     self.model_points,
        #     image_points,
        #     self.camera_matrix,
        #     self.dist_coeefs,
        #     rvec=self.r_vec,
        #     tvec=self.t_vec,
        #     useExtrinsicGuess=True)
        return rotation_vector, translation_vector

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
            # self.r_vec = rotation_vector
            # self.t_vec = translation_vector

        # (_, rotation_vector, translation_vector) = cv2.solvePnP(
        #     self.model_points_68,
        #     image_points,
        #     self.camera_matrix,
        #     self.dist_coeefs,
        #     rvec=rotation_vector,
        #     tvec=translation_vector,
        #     useExtrinsicGuess=True)

        return rotation_vector, translation_vector

    def draw_annotation_box(self, image, rotation_vector, translation_vector, color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def draw_axis(self, img, R, t):
        points = np.float32(
            [[30, 0, 0], [0, 30, 0], [0, 0, 30], [0, 0, 0]]).reshape(-1, 3)

        axisPoints, _ = cv2.projectPoints(
            points, R, t, self.camera_matrix, self.dist_coeefs)

        cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[0].ravel()), (255, 0, 0), 3)
        cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[1].ravel()), (0, 255, 0), 3)
        cv2.line(img, tuple(axisPoints[3].ravel()), tuple(
            axisPoints[2].ravel()), (0, 0, 255), 3)

    def draw_axes(self, img, R, t):
        cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 30)

    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 68 marks"""
        pose_marks = [marks[30], marks[8], marks[36], marks[45], marks[48], marks[54]]
        return pose_marks

    @staticmethod
    def get_euler(rotation_vector, translation_vector):
        """
        此函数用于从旋转向量计算欧拉角
        :param translation_vector: 输入为偏移向量
        :param rotation_vector: 输入为旋转向量
        :return: 返回欧拉角在三个轴上的值
        """
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        yaw = eulerAngles[1]
        pitch = eulerAngles[0]
        roll = eulerAngles[2]
        rot_params = np.array([yaw, pitch, roll])
        return rot_params
