import cv2
from local_feature import LocalFeature


class SURF(LocalFeature):

    def __init__(self,
                 image_size,
                 keypoint_image_border_size,
                 max_keypoint_count,
                 ldescriptor_length,
                 hessian_threshold,
                 extended,
                 upright):
        super(SURF, self).__init__(
            image_size=image_size,
            keypoint_image_border_size=keypoint_image_border_size,
            max_keypoint_count=max_keypoint_count,
            ldescriptor_length=ldescriptor_length)

        self.feature_detector = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            nOctaves=4,
            nOctaveLayers=3,
            extended=extended,
            upright=upright)
        self.descriptor_extractor = self.feature_detector
