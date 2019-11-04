# Kütüphane eklemeleri
import cv2
from os import listdir, getcwd, mkdir
from os.path import isfile, join, exists
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Dataset preparation for GAN input')

parser.add_argument('-r', '--raw-dir', action='store', nargs='?', default='raw_dir', dest='raw_dir', help="")
parser.add_argument('-o', '--output-dir', action='store', nargs='?', default='output_dir', dest='output_dir', help="")
parser.add_argument('-a', '--contrast', type=float, action='store', nargs='?', default=1, dest='contrast', help="")
parser.add_argument('-b', '--brightness', type=int, action='store', nargs='?', default=0, dest='brightness', help="")

parameters = parser.parse_args(['-r', 'raw_dir', '-o', 'output_dir', '-a', '0.1', '-b', '80'])

dir_path = getcwd()
dir_raw_img = join(dir_path, parameters.raw_dir)
dir_out_img = join(dir_path, parameters.output_dir)
dir_high_cont_img = join(dir_out_img, 'high')
dir_low_cont_img = join(dir_out_img, 'low')

if not exists(dir_out_img):
    mkdir(dir_out_img)
    mkdir(dir_high_cont_img)
    mkdir(dir_low_cont_img)

filename_raw = [join(parameters.raw_dir, f) for f in listdir(dir_raw_img) if isfile(join(dir_raw_img, f)) and f.endswith('.jpg')]

clh = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))


def img_contrast_edit(image):
    clahe_img = clh.apply(image)
    low_contr_image = cv2.convertScaleAbs(image, alpha=parameters.contrast, beta=parameters.brightness)
    return low_contr_image, clahe_img


j_sum = 0
for i, img_path in enumerate(filename_raw):
    j = 0
    if i % 100 == 0:
        print(f'\nImage {i} loading')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if img is None:
        continue
    sx, sy = img.shape
    if i % 100 == 0:
        print(f'Processing..')

    for x in range(0, sx - 256, 90):
        for y in range(0, sy - 256, 90):
            roi = img[x:x + 256, y:y + 256]
            low_cont, high_cont = img_contrast_edit(roi)
            cv2.imwrite(join(dir_high_cont_img, f'img{i:05d}_sub{j:03d}_high.jpeg'), high_cont)
            cv2.imwrite(join(dir_low_cont_img, f'img{i:05d}_sub{j:03d}_low.jpeg'), low_cont)
            j += 1
        j += 1

    j_sum += j
    if i % 100 == 0:
        print(f"Total {2 * j_sum} number sub-image(s) saved from {i} image(s)\n")
