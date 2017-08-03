from os import listdir
from os.path import join

from util import is_image, load_image

class DataLoaderHelper():
    def __init__(self, image_dir):
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image(x)]
        print(self.image_filenames)

    def __getitem__(self, index):
        input = load_image(join(self.a_path, self.image_filenames[index]))
        target = load_image(join(self.b_path, self.image_filenames[index]))

    def __len__(self):
        return len(self.image_filenames)


def get_training_data(root_dir):
    train_dir = join(root_dir, "train")
    return DataLoaderHelper(train_dir)

def get_test_data(root_dir):
    test_dir = join(root_dir, "test")
    return DataLoaderHelper(test_dir)
