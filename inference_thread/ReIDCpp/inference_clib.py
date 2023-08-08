import os
import cv2
import PIL.Image as Image
import tensorrt as trt
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import time
from ctypes import * 

from collections import OrderedDict

def preprocessing(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
    # image = image[:,:,::-1]
    # image = np.full((256,128,3), 255, dtype=np.uint8)
    # need CHW (RGB)
    return np.rollaxis(image, 2,0)

class ReidEngine():
    def __init__(self,trt_engine_path):
        # trt_engine_path = '/home/xujiahong/NX/ReIDFsid_multithread/ThirdParty/ReIDFsidEngine/fsid-res50-INT8.engine'
        libFastRTPython = cdll.LoadLibrary("/home/xujiahong/NX/ReIDFsid_multithread/ThirdParty/ReIDCpp/libFastRTPython.so")

        self.buildFeatEX = libFastRTPython.buildFeatEX
        self.buildFeatEX.argtypes = [c_char_p]
        self.buildFeatEX.restype = c_int

        self.FeatEX = libFastRTPython.FeatEX
        self.FeatEX.argtypes = [POINTER(c_char), c_int]
        self.FeatEX.restype = POINTER(c_float)

        # self.clean = libFastRTPython.clean
        # self.clean.restype = c_int
        ## initialize engine
        self.buildFeatEX(trt_engine_path.encode('ascii'))
        self.float_ptr = (c_float*128)()
        
    def preprocessing(self, image):
        # image = cv2.imread(img_path)
        image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
        return np.rollaxis(image, 2,0)

    def run_reid(self, img): 
        ## input img shape (C, H, W) BGR
        data_ctypes_ptr = img.ctypes.data_as(POINTER(c_char))
        feat_cbyte = self.FeatEX(data_ctypes_ptr, 1, self.float_ptr)
        feat = np.frombuffer(self.float_ptr, dtype=np.float32)
        feat_np = feat.copy() # 需要copy,否则会随着输入图片更新而改变
        return feat_np

    # def destuctor(self):
    #     self.clean
    #     return 



if __name__ == "__main__":
    ######### 
    # build eninge 
    data = cv2.imread('/home/xujiahong/NX/ReIDFsid_multithread_v2/persons.jpg')
    engine = ReidEngine('/home/xujiahong/NX/ReIDFsid_multithread_v2/ThirdParty/ReIDFsidEngine/fsid-res50-INT8.engine')

    import time
    from tqdm import tqdm
    start = time.time()
    l = range(1)
    for i in tqdm(l):
        input= engine.preprocessing(data)
        feat_np = engine.run_reid(input)
        print(feat_np)
    print('total time: ', time.time() - start)
    print('fps:{}'.format( 10000/(time.time()-start)))
    print(feat_np)
    # engine.clean
