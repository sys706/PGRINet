import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(img_names_file, img_files, mask_files):
    img_names = np.load(img_names_file)
    img_paths, mask_paths, img_size = make_dataset_dir(img_names, img_files, mask_files)

    return img_paths, mask_paths, img_size

def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths, len(img_paths)

def make_dataset_dir(img_names, img_dir, mask_dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []
    mask_paths = []

    for name in img_names:
        img_paths.append(os.path.join(img_dir, name))
        mask_paths.append(os.path.join(mask_dir, name))

    return img_paths, mask_paths, len(img_paths)
