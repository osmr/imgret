import cv2
import numpy as np


class VLAD(object):

    def __init__(self, lcluster_centers):
        super(VLAD, self).__init__()
        self.lcluster_centers = lcluster_centers
        self.gdescriptor_length = self.lcluster_centers.shape[0] * self.lcluster_centers.shape[1]
        self.lcluster_matcher = cv2.BFMatcher()

    def calc_descriptor(self, ldescriptors):
        assert (ldescriptors.shape[1] == self.lcluster_centers.shape[1])
        matches = self.lcluster_matcher.match(
            queryDescriptors=ldescriptors,
            trainDescriptors=self.lcluster_centers)
        vlad = np.zeros_like(self.lcluster_centers)
        for i, match in enumerate(matches):
            ldescriptor_idx = match.queryIdx
            lcluster_idx = match.trainIdx
            assert (ldescriptor_idx == i)
            vlad[lcluster_idx, :] += (ldescriptors[ldescriptor_idx, :] - self.lcluster_centers[lcluster_idx, :])
        vlad = vlad.flatten()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        vlad_norm = np.linalg.norm(vlad)
        assert (vlad_norm > 1e-5)
        # if vlad_norm > 1e-5:
        vlad /= vlad_norm
        return vlad

    @staticmethod
    def calc_lcluster_centers(ldescriptors,
                              ldescriptor_cluster_count=256,
                              kmeans_attempts=5):
        compactness, labels, centers = cv2.kmeans(
            data=ldescriptors,
            K=ldescriptor_cluster_count,
            bestLabels=None,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.001),
            attempts=kmeans_attempts,
            flags=cv2.KMEANS_PP_CENTERS)
        return centers
