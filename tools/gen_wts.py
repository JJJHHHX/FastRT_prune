# encoding: utf-8

import sys
import time
import struct
import argparse
sys.path.append('.')

import torch
# import torchvision
#from torchsummary import summary


def get_parser():
    parser = argparse.ArgumentParser(description="Encode pytorch weights for tensorrt.")
    parser.add_argument(
        "--prune",
        type= bool, 
        default=True,
        help='if the model is pruned'
    )
    parser.add_argument(
        "--pth_path",
        # default='/home/xujiahong/trt_engine/FastRT_pruned/pruned_model.pth',
        default='/home/xujiahong/fast-reid-master/logs/market1501/bagtricks_R50_prune_finetune/model_final.pth',
        help='path to pth model'
    )
    parser.add_argument(
        "--wts_path",
        default='./prune_0.5.wts',
        help='path to save tensorrt weights file(.wts)'
    )
    parser.add_argument(
        "--shape_path",
        default='./pruned_conv_output_shape_0.5.txt',
        help='conv output shape save to txt'
    )
    return parser

def gen_wts(args):
    """
        Thanks to https://github.com/wang-xinyu/tensorrtx
    """
    model_dict = torch.load(args.pth_path)["model"]

    
    print("Wait for it: {} ...".format(args.wts_path))
    f = open(args.wts_path, 'w')
    f.write("{}\n".format(len(model_dict.keys())))
    for k,v in model_dict.items():
        print('key: ', k)
        print('value: ', v.shape)     
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

def gen_conv_out_shape_wts(args):
    """
        Thanks to https://github.com/wang-xinyu/tensorrtx
    """
    model_dict = torch.load(args.pth_path)["model"]
    
    print("Wait for it: {} ...".format(args.wts_path))
    keys = list(model_dict.keys())
    save_keys = []
    for k in keys:
        if k.endswith('conv1.weight') or k.endswith("conv2.weight"):
            save_keys.append(k)

    f = open(args.shape_path, 'w')
    for k in save_keys:
        weight = model_dict[k]
        print('key: ', k)
        print('value: ', weight.shape) 
        v =  weight.shape[0]
        print('saved values: ', v)
        f.write("{} {}".format(k, v))
        f.write("\n")
        
if __name__ == '__main__':
    args = get_parser().parse_args()
    gen_wts(args)
    if args.prune:
        gen_conv_out_shape_wts(args)    
    