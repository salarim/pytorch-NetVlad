import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np
from os.path import join, exists
from scipy.io import loadmat

from collections import namedtuple
from PIL import Image


root_dir = '../datasets/InLoc'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to InLoc dataset')

queries_dir = join(root_dir, 'iphone7')

def input_transform():
    return transforms.Compose([
        transforms.Resize((640,480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def get_whole_test_set():
    return WholeDatasetFromStruct(input_transform=input_transform())

dbStruct = namedtuple('dbStruct', ['dbImage', 'qImage', 'numDb', 'numQ'])

def parse_dbStruct():
    cutout_path = join(root_dir, 'cutout_imgnames_all.npy')
    query_path = join(root_dir, 'query_imgnames_all.npy')

    cutout_imgnames_all = np.load(cutout_path, allow_pickle=True)
    query_imgnames_all = np.load(query_path, allow_pickle=True)

    dbImage = cutout_imgnames_all.tolist()

    qImage = query_imgnames_all.tolist()

    numDb = len(dbImage)
    numQ = len(qImage)

    return dbStruct(dbImage, qImage, numDb, numQ)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform

        self.dbStruct = parse_dbStruct()
        self.images = [join(root_dir, 'cutouts_imageonly', dbIm[0]) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm[0]) for qIm in self.dbStruct.qImage]


        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        return None
        
if __name__ == "__main__":
    whole_test_set = get_whole_test_set()
    print(whole_test_set[10][0].shape, whole_test_set[10][1])
