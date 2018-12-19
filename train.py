import argparse
import os
import re
import fnmatch
import numpy as np
from .sift import SIFT
from .surf import SURF
from .vlad import VLAD
from .pca_global_descriptor import PCAGlobalDescriptor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training a model for image retrieval',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data',
        type=str,
        default='../imgret_data/',
        help='path to directory with training images')
    parser.add_argument(
        '--local-feature',
        type=str,
        default='SIFT',
        help='type of local feature. options are SIFT and SURF.')
    parser.add_argument(
        '--out-dir-path',
        type=str,
        default='../imgret_data/',
        help='path to directory for output model file')
    parser.add_argument(
        '--model-file-name',
        type=str,
        default='model.npz',
        help='resulted model file name')
    args = parser.parse_args()
    return args


def get_test_image_file_paths(train_image_dir_path):
    image_file_names = fnmatch.filter(os.listdir(train_image_dir_path), '*.jpg')
    image_file_names = [n for n in image_file_names if os.path.isfile(os.path.join(train_image_dir_path, n))]
    image_file_names.sort(
        key=lambda var: ['{:10}'.format(int(x)) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    image_file_paths = [os.path.join(train_image_dir_path, n) for n in image_file_names]
    return image_file_paths


def calc_lcluster_centers(train_image_dir_path,
                          local_feature):
    image_file_paths = get_test_image_file_paths(train_image_dir_path)
    ldescriptors_list = local_feature.calc_descriptors_list(image_file_paths)
    ldescriptors = np.vstack(ldescriptors_list)
    lcluster_centers = VLAD.calc_lcluster_centers(ldescriptors)
    return lcluster_centers


def calc_pca(train_image_dir_path,
             local_feature,
             global_feature):
    image_file_paths = get_test_image_file_paths(train_image_dir_path)
    ldescriptors_list = local_feature.calc_descriptors_list(image_file_paths)
    gdescriptor_list = []
    for ldescriptors in ldescriptors_list:
        gdescriptor = global_feature.calc_descriptor(ldescriptors=ldescriptors)
        gdescriptor_list += [gdescriptor]
    gdescriptors = np.vstack(gdescriptor_list)
    pca_mean, pca_eigenvectors = PCAGlobalDescriptor.calc_pca(
        gdescriptors=gdescriptors,
        pca_length=256)
    return pca_mean, pca_eigenvectors


def main():
    args = parse_args()

    train_image_dir_path = args.data

    model = {}

    if args.local_feature == "SIFT":
        local_feature = SIFT()
        model["local_feature"] = np.array(["SIFT"])
    elif args.local_feature == "SURF":
        local_feature = SURF()
        model["local_feature"] = np.array(["SIFT"])
    else:
        raise ValueError("local_feature")

    lcluster_centers = calc_lcluster_centers(
        train_image_dir_path=train_image_dir_path,
        local_feature=local_feature)
    model["lcluster_centers"] = lcluster_centers

    global_feature = VLAD(lcluster_centers=lcluster_centers)

    pca_mean, pca_eigenvectors = calc_pca(
        train_image_dir_path=train_image_dir_path,
        local_feature=local_feature,
        global_feature=global_feature)
    model["pca_mean"] = pca_mean
    model["pca_eigenvectors"] = pca_eigenvectors

    model_file_path = os.path.join(args.out_dir_path, args.model_file_name)
    np.savez_compressed(model_file_path, **model)


if __name__ == '__main__':
    main()