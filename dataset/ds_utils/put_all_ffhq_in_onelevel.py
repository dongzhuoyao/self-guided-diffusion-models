import glob
import os
import shutil
from diffusion_utils.util import make_clean_dir
from tqdm import tqdm

ffhq_train_root = '/home/thu/data/ffhq/thumbnails64x64'
dst_dir = '/home/thu/data/ffhq/thumbnails64x64_onelevel'

make_clean_dir(dst_dir)

for full_file_path in tqdm(glob.iglob(ffhq_train_root + '**/**', recursive=True)):
    if os.path.isfile(full_file_path) and full_file_path.endswith('.png'):
        print(full_file_path)
        shutil.copyfile(full_file_path, os.path.join(
            dst_dir, os.path.basename(full_file_path)))



