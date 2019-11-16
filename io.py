from os import listdir, getcwd
from os.path import join, isfile
import argparse


parser = argparse.ArgumentParser(description='Dataset preparation for GAN input')

parser.add_argument('-i', '--input-dir', action='store', nargs='?', default='output_dir', dest='input_dir', help=" ")
parser.add_argument('-g', '--gan-output', action='store', nargs='?', default='gan_out_dir', dest='gan_out_dir')
parameters = parser.parse_args(['-i', 'output_dir', '-g', 'gan_out_dir'])

dir_out_img = join(join(getcwd(), 'data_preprocess'), parameters.input_dir)
dir_high_cont = join('data_preprocess', join(parameters.input_dir, 'high'))
dir_low_cont = join('data_preprocess', join(parameters.input_dir, 'low'))

high_cont_file = [join(dir_high_cont, f) for f in listdir(dir_high_cont) if isfile(join(dir_high_cont, f))]
low_cont_file = [join(dir_low_cont, f) for f in listdir(dir_low_cont) if isfile(join(dir_low_cont, f))]

