import cv2


class PCAGlobalDescriptor(object):

    def __init__(self,
                 local_feature,
                 global_feature,
                 pca_mean,
                 pca_eigenvectors):
        super(PCAGlobalDescriptor, self).__init__()
        self.local_feature = local_feature
        self.global_feature = global_feature
        self.pca_mean = pca_mean
        self.pca_eigenvectors = pca_eigenvectors

        self.gdescriptor_length = pca_eigenvectors.shape[0]

    def calc_descriptor(self, image):
        ldescriptors = self.local_feature.calc_descriptors(image=image)
        if ldescriptors is None:
            return None
        raw_gdescriptor = self.global_feature.calc_descriptor(ldescriptors=ldescriptors)
        pca_gdescriptor = cv2.PCAProject(
            data=raw_gdescriptor.reshape(1, -1),
            mean=self.pca_mean,
            eigenvectors=self.pca_eigenvectors)
        return pca_gdescriptor.flatten()

    @staticmethod
    def calc_pca(gdescriptors,
                 pca_length=256):
        mean, eigenvectors = cv2.PCACompute(
            data=gdescriptors,
            mean=None,
            maxComponents=pca_length)
        return mean, eigenvectors
