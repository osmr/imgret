import os
import cv2


class LocalFeature(object):

    def __init__(self,
                 image_size=(640, 360),
                 keypoint_image_border_size=10,
                 max_keypoint_count=512,
                 ldescriptor_length=128):
        super(LocalFeature, self).__init__()
        self.image_size = image_size
        self.keypoint_image_border_size = keypoint_image_border_size
        self.max_keypoint_count = max_keypoint_count
        self.ldescriptor_length = ldescriptor_length

        self.feature_detector = None
        self.descriptor_extractor = None

    def calc_descriptors(self, image):
        if self.feature_detector is None:
            raise NotImplementedError()
        if (len(image.shape) == 3) and (image.shape[2] == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        assert (len(image.shape) == 2)
        if image.shape != self.image_size[::-1]:
            image = cv2.resize(image, self.image_size)
        cv2.equalizeHist(image, image)
        keypoints = self._detect_keypoints(image)
        if len(keypoints) == 0:
            return None
        _, ldescriptors = self.descriptor_extractor.compute(
            image=image,
            keypoints=keypoints)
        assert (ldescriptors.shape[1] == self.ldescriptor_length)
        return ldescriptors

    def calc_descriptors_list(self, image_file_paths):
        ldescriptors_list = []
        for image_file_path in image_file_paths:
            assert os.path.exists(image_file_path)
            image = cv2.imread(filename=image_file_path, flags=0)
            image = cv2.resize(image, self.image_size)
            ldescriptors = self.calc_descriptors(image=image)
            if ldescriptors is None:
                continue
            ldescriptors_list += [ldescriptors]
        return ldescriptors_list

    def _detect_keypoints(self, image):
        keypoints = self.feature_detector.detect(image)
        keypoints = self._cv2_run_by_image_border(
            keypoints=keypoints,
            image_size=self.image_size,
            border_size=self.keypoint_image_border_size)
        keypoints.sort(key=lambda kp: kp.response, reverse=True)
        return keypoints[:self.max_keypoint_count]

    @staticmethod
    def _cv2_run_by_image_border(keypoints,
                                 image_size,
                                 border_size):
        border_rect = (border_size,
                       border_size,
                       image_size[0] - 2 * border_size,
                       image_size[1] - 2 * border_size)
        keypoints = [kp for kp in keypoints if LocalFeature._cv2_rect_contains(border_rect, kp.pt)]
        return keypoints

    @staticmethod
    def _cv2_rect_contains(rect, pt):
        return rect[0] < pt[0] < rect[0] + rect[2] and rect[1] < pt[1] < rect[1] + rect[3]
