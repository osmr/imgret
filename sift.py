import cv2
from local_feature import LocalFeature


class SIFT(LocalFeature):

    def __init__(self,
                 image_size,
                 keypoint_image_border_size,
                 max_keypoint_count,
                 ldescriptor_length,
                 contrast_threshold,
                 edge_threshold):
        super(SIFT, self).__init__(
            image_size=image_size,
            keypoint_image_border_size=keypoint_image_border_size,
            max_keypoint_count=max_keypoint_count,
            ldescriptor_length=ldescriptor_length)

        self.feature_detector = cv2.xfeatures2d.SIFT_create(
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold)
        self.descriptor_extractor = self.feature_detector
