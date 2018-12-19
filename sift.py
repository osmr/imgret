import cv2
from .local_feature import LocalFeature


class SIFT(LocalFeature):

    def __init__(self,
                 image_size=(640, 360),
                 keypoint_image_border_size=10,
                 max_keypoint_count=512,
                 ldescriptor_length=128,
                 contrast_threshold=0.04,
                 edge_threshold=10):
        super(SIFT, self).__init__(
            image_size=image_size,
            keypoint_image_border_size=keypoint_image_border_size,
            max_keypoint_count=max_keypoint_count,
            ldescriptor_length=ldescriptor_length)

        self.feature_detector = cv2.xfeatures2d.SIFT_create(
            contrast_threshold=contrast_threshold,
            edge_threshold=edge_threshold)
        self.descriptor_extractor = self.feature_detector
