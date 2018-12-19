import cv2
from local_feature import LocalFeature


class VGG(LocalFeature):

    def __init__(self,
                 image_size,
                 keypoint_image_border_size,
                 max_keypoint_count,
                 ldescriptor_length,
                 use_scale_orientation):
        super(VGG, self).__init__(
            image_size=image_size,
            keypoint_image_border_size=keypoint_image_border_size,
            max_keypoint_count=max_keypoint_count,
            ldescriptor_length=ldescriptor_length)

        self.feature_detector = cv2.FastFeatureDetector_create()
        self.descriptor_extractor = cv2.xfeatures2d.VGG_create(
            use_scale_orientation=use_scale_orientation)
