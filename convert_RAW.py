import os
import numpy as np
import rawpy
import glob
from matplotlib import pyplot as plt
import multiprocessing

input_dir = './dataset/Sony/short/'
output_dir = './dataset/Sony/short/JPEG'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

im_sets = glob.glob(input_dir + '*.ARW')

# for raw_file in im_sets:
#     print(os.path.basename(raw_file))
#     raw_map = rawpy.imread(raw_file)
#     im = raw_map.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
#     im = np.expand_dims(np.float32(im / 65535.0), axis=0)
#     plt.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(raw_file))[0]+'.png'), im.squeeze(0))


def convert_raw(raw_file):
    print(os.path.basename(raw_file))
    raw_map = rawpy.imread(raw_file)
    im = raw_map.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    im = np.expand_dims(np.float32(im / 65535.0), axis=0)
    plt.imsave(os.path.join(output_dir, os.path.splitext(os.path.basename(raw_file))[0] + '.png'), im.squeeze(0))


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=16)
    pool.map(convert_raw, im_sets)
    pool.close()
    pool.join()

