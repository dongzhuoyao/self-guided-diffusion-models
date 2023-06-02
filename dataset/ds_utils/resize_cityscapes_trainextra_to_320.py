
import glob
import imp
import os
from re import S
import shutil
from tqdm import tqdm
from diffusion_utils.util import make_clean_dir
from PIL import Image
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

src_dir = '/home/thu/data/stego_data/cityscapes'
target_dir = '/home/thu/data/sg_data/cs320_v2'


target_img_size = 320


cs_train_extra_img_dir = os.path.join(src_dir, 'leftImg8bit/train_extra')
cs_train_extra_label_dir = os.path.join(src_dir, 'gtCoarse/train_extra')

cs_train_extra_img_dir_320 = os.path.join(target_dir, 'train_extra_images')
cs_train_extra_label_dir_320 = os.path.join(target_dir, 'train_extra_labels')

#######

cs_val_img_dir = os.path.join(src_dir, 'leftImg8bit/val')
cs_val_label_dir = os.path.join(src_dir, 'gtCoarse/val')

cs_val_img_dir_320 = os.path.join(target_dir, 'val_images')
cs_val_label_dir_320 = os.path.join(target_dir, 'val_labels')


class RecursiveDirDataset(Dataset):
    def __init__(self, src_img_dir, target_img_dir, interp=Image.BILINEAR):
        self.src_img_dir = src_img_dir
        self.target_img_dir = target_img_dir
        self.interp = interp

        self.file_names = list()
        for full_file_path in glob.iglob(src_img_dir + '**/**', recursive=True):
            if os.path.isfile(full_file_path) and full_file_path.endswith('.png'):
                self.file_names.append(full_file_path)
        assert len(self.file_names) > 0, f"no file found in {src_img_dir}"
        logger.warning(f"{len(self.file_names)} files found in {src_img_dir}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.src_img_dir, self.file_names[idx])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((target_img_size, target_img_size), self.interp)
        image.save(os.path.join(self.target_img_dir,
                   os.path.basename(img_path)))
        return 0


def resize_and_save_real(src_dir,  target_dir,  batch_size, interp=Image.BILINEAR):
    make_clean_dir(target_dir)
    ds = DataLoader(RecursiveDirDataset(src_img_dir=src_dir,
                    target_img_dir=target_dir, interp=interp), batch_size=batch_size, num_workers=8)
    for data in tqdm(ds, total=len(ds)):
        pass


def resize_and_save(src_img_dir, src_label_dir, target_img_dir, target_label_dir,  batch_size=256):
    resize_and_save_real(src_img_dir, target_img_dir,
                         batch_size, interp=Image.BILINEAR)
    resize_and_save_real(src_label_dir, target_label_dir,
                         batch_size, interp=Image.NEAREST)


resize_and_save(src_img_dir=cs_train_extra_img_dir, src_label_dir=cs_train_extra_label_dir,
                target_img_dir=cs_train_extra_img_dir_320, target_label_dir=cs_train_extra_label_dir_320)


resize_and_save(src_img_dir=cs_val_img_dir, src_label_dir=cs_val_label_dir,
                target_img_dir=cs_val_img_dir_320, target_label_dir=cs_val_label_dir_320)
