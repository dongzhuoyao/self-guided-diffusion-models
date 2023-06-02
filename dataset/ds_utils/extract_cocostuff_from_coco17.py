import os
from pathlib import Path
import shutil
from diffusion_utils.util import make_clean_dir
from tqdm import tqdm

is_debug = False
coco17_root_dir = '/home/thu/data/stego_data/cocostuff/images'
coco17_train_dir = os.path.join(coco17_root_dir, 'train2017')
coco17_val_dir = os.path.join(coco17_root_dir, 'val2017')

cocostuff_train_txt = '/home/thu/data/stego_data/cocostuff/curated/train2017/Coco164kFull_Stuff_Coarse_7.txt'
cocostuff_val_txt = '/home/thu/data/stego_data/cocostuff/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt'

cocostuff27_dest_dir = '~/data/cocostuff27/images'
cocostuff27_dest_dir = str(Path(cocostuff27_dest_dir).expanduser().resolve())
cocostuff27_train_dir = os.path.join(cocostuff27_dest_dir, 'train')
cocostuff27_val_dir = os.path.join(cocostuff27_dest_dir, 'val')

make_clean_dir(cocostuff27_train_dir)
make_clean_dir(cocostuff27_val_dir)


with open(cocostuff_train_txt, 'r') as f:
    cocostuff_train_list = f.readlines()
    cocostuff_train_list = [x.strip() for x in cocostuff_train_list]
with open(cocostuff_val_txt, 'r') as f:
    cocostuff_val_list = f.readlines()
    cocostuff_val_list = [x.strip() for x in cocostuff_val_list]

if is_debug:
    cocostuff_train_list = cocostuff_train_list[:100]
    cocostuff_val_list = cocostuff_val_list[:100]

for train_id in tqdm(cocostuff_train_list, total=len(cocostuff_train_list)):
    shutil.copyfile(os.path.join(coco17_train_dir, f'{train_id}.jpg'), os.path.join(
        cocostuff27_train_dir, f'{train_id}.jpg'))
print('cocostuff27_train_dir', {len(os.listdir(cocostuff27_train_dir))})

for val_id in tqdm(cocostuff_val_list, total=len(cocostuff_val_list)):
    shutil.copyfile(os.path.join(coco17_val_dir, f'{val_id}.jpg'), os.path.join(
        cocostuff27_val_dir, f'{val_id}.jpg'))
print('cocostuff27_val_dir', {len(os.listdir(cocostuff27_val_dir))})
