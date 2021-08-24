import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=7, help='segmentation classes')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        parser.add_argument('--model_path', type=str, help='path to store trained model files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files, no need to include task specific name')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3], help='mode for training procedure. 1.fcn 2.fcn+1 3.fcn+2')
        parser.add_argument('--dataset', type=int, default=2, choices=[1, 2], help='dataset for training procedure. 1.deep 2.IA')
        parser.add_argument('--train', action='store_true', default=False, help='train')
        parser.add_argument('--val', action='store_true', default=False, help='val')
        parser.add_argument('--context10', type=int, default=2, help='context10')
        parser.add_argument('--context15', type=int, default=3, help='context15')
        parser.add_argument('--pre_path', type=str, default="", help='name for pre model path')
        parser.add_argument('--glo_path_10', type=str, default="", help='name for medium model path')
        parser.add_argument('--glo_path_15', type=str, default="", help='name for large model path')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--sub_batch_size', type=int, default=6, help='batch size for using local image patches')
        parser.add_argument('--size_p', type=int, default=508, help='size (in pixel) for cropped local image')
        parser.add_argument('--size_g', type=int, default=508, help='size (in pixel) for resized global image')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.num_epochs = 100
        args.start = 50
        args.lens = 50
        args.lr = 5e-5
        return args
