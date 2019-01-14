import io
import os
from glob import glob

import numpy as np
import tensorflow as tf
from imageio import imread, imwrite
from tqdm import tqdm
import random

from image_segmentation import build_data

def image_array_to_bytes(array):
    byte_buffer = io.BytesIO()
    imwrite(byte_buffer, array, "png")
    byte_buffer.seek(0)
    return byte_buffer.read()

def get_segmentation_indices(path):
    img = imread(path)
    index_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    index_img[img[:, :, 0] > 0, 0] = 1
    return image_array_to_bytes(index_img)

def main():
    width, height = 512, 512
    color_paths = sorted(glob("/home/joel/datasets/ade_mrpt/*_C.png"))
    seg_paths = sorted(glob("/home/joel/datasets/ade_mrpt/*_S.png"))

    assert len(color_paths) == len(seg_paths)

    # Shuffle the paths randomly
    paths = list(zip(color_paths, seg_paths))
    random.shuffle(paths)
    
    with tf.python_io.TFRecordWriter("dataset.tfrecords") as tfrecord_writer:
        for color_path, seg_path in tqdm(paths):
            image_data = tf.gfile.FastGFile(color_path, "rb").read()
            seg_data = get_segmentation_indices(seg_path)

            example = build_data.image_seg_to_tfexample(
                    image_data, color_path, height, width, seg_data)
            tfrecord_writer.write(example.SerializeToString())

    print("Done")

if __name__ == "__main__":
    main()
