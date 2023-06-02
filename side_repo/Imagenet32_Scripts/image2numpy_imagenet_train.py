# http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10/35034287

from argparse import ArgumentParser
from utils import *
import os
from scipy import misc
import numpy as np
import imageio

def parse_arguments():
    # python image2numpy_imagenet_train.py --in_dir /home/thu/data/imagenet_official_train_resized32/box --out_dir /home/thu/data/imagenet_official_np/size32/train

    # python image2numpy_imagenet_train.py --in_dir /home/thu/data/imagenet_official_train_resized64/box --out_dir /home/thu/data/imagenet_official_np/size64/train

    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--in_dir', default='/home/thu/data/imagenet_train_resized32/lanczos', help="Input directory with source images")
    parser.add_argument('-o', '--out_dir', default='/home/thu/data/imagenet_train_resized32_np',
                        help="Output directory for pickle files")
    args = parser.parse_args()

    return args.in_dir, args.out_dir


# Strong assumption about in_dir and out_dir (They must contain proper data)
def process_folder(in_dir, out_dir):
    label_dict = get_label_dict()
    folders = get_ordered_folders()

    # Here subsample folders (If desired) [1*]
    # folders = folders[0::10]
    # folders = folders[900:902]

    print("Processing folder %s" % in_dir)
    data_list_train = []
    labels_list_train = []
    image_name_train = []

    for f_id, folder in enumerate(folders):
        label = label_dict[folder]
        print("Processing images from folder %s as label %d" % (folder, label))
        # Get images from this folder
        images = []
        image_names = []
        for image_name in os.listdir(os.path.join(in_dir, folder)):
            full_path = os.path.join(in_dir, folder, image_name)
            try:
                img = imageio.imread(full_path)
                r = img[:, :, 0].flatten()
                g = img[:, :, 1].flatten()
                b = img[:, :, 2].flatten()

                arr = np.array(list(r) + list(g) + list(b), dtype=np.uint8)

                images.append(arr)
                image_names.append(image_name)
            except:
                print('Cant process image %s' % full_path)
                with open("log_img2np.txt", "a") as f:
                    f.write("Couldn't read: %s \n" %
                            os.path.join(in_dir, image_name))
                continue
            

        data = np.row_stack(images)
        samples_num = data.shape[0]
        labels = [label] * samples_num

        labels_list_train.extend(labels)
        data_list_train.append(data)
        image_name_train.extend(image_names)

        print('Label: %d: %s has %d samples' % (label, folder, samples_num))
        #if f_id>10:break

    x = np.concatenate(data_list_train, axis=0)
    y = labels_list_train
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # If you subsample folders [1*] this will not compute mean over all training images
    x_mean = np.mean(x, axis=0)

    # Shuffled indices
    train_indices = np.arange(x.shape[0])
    np.random.shuffle(train_indices)

    curr_index = 0
    size = x.shape[0] // 10

    # Create first 9 files
    y_test = []
    for i in range(1, 10):
        d = {
            'data': x[train_indices[curr_index: (curr_index + size)], :],
            'labels': np.array(y)[train_indices[curr_index: (curr_index + size)]].tolist(),
            'mean': x_mean,
            'names': [image_name_train[_] for _ in train_indices[curr_index: (curr_index + size)]],
        }
        pickle.dump(d, open(os.path.join(
            out_dir, 'train_data_batch_%d' % i), 'wb'))
        curr_index += size
        y_test.extend(d['labels'])

    # Create last file
    d = {
        'data': x[train_indices[curr_index:], :],
        'labels': np.array(y)[train_indices[curr_index:]].tolist(),
        'mean': x_mean,
        'names': [image_name_train[_] for _ in train_indices[curr_index:]],
    }
    pickle.dump(d, open(os.path.join(out_dir, 'train_data_batch_10'), 'wb'))

    y_test.extend(d['labels'])

    count = np.zeros([1000])

    for i in y_test:
        count[i-1] += 1

    for i in range(1000):
        print('%d : %d' % (i, count[i]))

    print('SUM: %d' % len(y_test))


if __name__ == '__main__':
    in_dir, out_dir = parse_arguments()

    print("Start program ...")
    process_folder(in_dir=in_dir, out_dir=out_dir)
    print("Finished.")
