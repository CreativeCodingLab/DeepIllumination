from os import listdir
from os.path import join

import torch.utils.data as data

from util import is_image, load_image

class DataLoaderHelper(data.Dataset):
    def __init__(self, image_dir):
        super(DataLoaderHelper, self).__init__()
        self.albedo_path = join(image_dir, "albedo")
        self.depth_path = join(image_dir, "depth")
        self.direct_path = join(image_dir, "direct")
        self.normal_path = join(image_dir, "normal")
        self.gt_path = join(image_dir, "gt")
        self.image_filenames = [x for x in listdir(self.albedo_path) if is_image(x)]


    def __getitem__(self, index):
        albedo = load_image(join(self.albedo_path, self.image_filenames[index]))
        depth = load_image(join(self.depth_path, self.image_filenames[index]))
        direct = load_image(join(self.direct_path, self.image_filenames[index]))
        normal = load_image(join(self.normal_path, self.image_filenames[index]))
        gt = load_image(join(self.gt_path, self.image_filenames[index]))
        return albedo, direct, normal, depth, gt

    def __len__(self):
        return len(self.image_filenames)
