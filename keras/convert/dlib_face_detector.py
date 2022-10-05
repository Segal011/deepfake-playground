import warnings
import cv2
import dlib
import torch
import io
import numpy as np
from imutils import face_utils


# from imutils import face_detection


class DlibDetectedFace:
    def __init__(self):
        super().__init__()

        warnings.warn('Warning: this detector is deprecated. Please use a different one, i.e.: S3FD.')

        self.detector = dlib.get_frontal_face_detector() # TODO pabandyt ir cnn_face_detection_model
        self.predictor = dlib.shape_predictor(
            r"C:\Users\37060\Documents\GitHub\magistras\deepfake-playground\shape_predictor_68_face_landmarks.dat")

    # @staticmethod
    # def tensor_or_path_to_ndarray(tensor_or_path):
    #     """Convert path (represented as a string) or torch.tensor to a numpy.ndarray
    #     Arguments:
    #         tensor_or_path {numpy.ndarray, torch.tensor or string} -- path to the image, or the image itself
    #     """
    #     if isinstance(tensor_or_path, str):
    #         return io.imread(tensor_or_path)
    #     elif torch.is_tensor(tensor_or_path):
    #         return tensor_or_path.cpu().numpy()
    #     elif isinstance(tensor_or_path, np.ndarray):
    #         return tensor_or_path
    #     else:
    #         raise TypeError
    def parse_parts(self, landmarks):
        """ Extended face hull mask """
        # mid points between the side of face and eye point
        ml_pnt = (landmarks[36] + landmarks[0]) // 2
        mr_pnt = (landmarks[16] + landmarks[45]) // 2

        # mid points between the mid points and eye
        ql_pnt = (landmarks[36] + ml_pnt) // 2
        qr_pnt = (landmarks[45] + mr_pnt) // 2

        # Top of the eye arrays
        bot_l = np.array((ql_pnt, landmarks[36], landmarks[37], landmarks[38], landmarks[39]))
        bot_r = np.array((landmarks[42], landmarks[43], landmarks[44], landmarks[45], qr_pnt))

        # Eyebrow arrays
        top_l = landmarks[17:22]
        top_r = landmarks[22:27]

        # Adjust eyebrow arrays
        landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

        r_jaw = (landmarks[0:9], landmarks[17:18])
        l_jaw = (landmarks[8:17], landmarks[26:27])
        r_cheek = (landmarks[17:20], landmarks[8:9])
        l_cheek = (landmarks[24:27], landmarks[8:9])
        nose_ridge = (landmarks[19:25], landmarks[8:9],)
        r_eye = (landmarks[17:22],
                 landmarks[27:28],
                 landmarks[31:36],
                 landmarks[8:9])
        l_eye = (landmarks[22:27],
                 landmarks[27:28],
                 landmarks[31:36],
                 landmarks[8:9])
        nose = (landmarks[27:31], landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]
        return parts

    def detect_from_image(self, image):
        # image = self.tensor_or_path_to_ndarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detected_faces = self.detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        found_faces = []
        for face in detected_faces:
            shape = self.predictor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), face)

            mask = np.zeros(image.shape[:2], np.uint8)
            mask_shape = face_utils.shape_to_np(shape)
            mask_shape = cv2.convexHull(mask_shape)
            cv2.drawContours(mask, [mask_shape], -1, 255, -1)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)

            detected_face = dict(
                shape=[face.left(), face.top(), face.right(), face.bottom()],
                landmarks=[(shape.part(n).x, shape.part(n).y) for n in range(0, 68)],
                mask=mask
            )
            found_faces.append(detected_face)

        """ Run model to get predictions """


        return found_faces[0]
