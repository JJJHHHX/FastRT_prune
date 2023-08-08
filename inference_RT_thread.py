import cv2
import tensorrt as trt
import numpy as np
# import logging
# from tqdm import tqdm
import threading
import datetime
import time

import pycuda.driver as cuda
# import pycuda.autoinit

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()


def preprocessing_person(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
    image = image[:,:,::-1]
    ######## image = np.full((256,128,3), 255, dtype=np.uint8)
    #### need CHW (BGR)
    return np.rollaxis(image, 2,0)

def preprocessing_car(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    image = image[:,:,::-1]
    ######## image = np.full((256,128,3), 255, dtype=np.uint8)
    #### need CHW (BGR)
    return np.rollaxis(image, 2,0)


class FeatureExtractor:
    
    def __init__(self,engine_path,max_batch_size=1):
        ### 多进程初始化方法
        cuda.init()
        self.cfx= cuda.Device(0).make_context()
        
        self.engine_path = engine_path
        self.batch_size = max_batch_size
        self.logger = trt.Logger( trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

        self.cfx.pop()
            
    @staticmethod
    def load_engine(trt_runtime, engine_path):           
        with open(engine_path, 'rb') as f, trt_runtime:
            return trt_runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self):
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs=HostDeviceMem(host_mem, device_mem)
            else:
                outputs=HostDeviceMem(host_mem, device_mem)
        
        return inputs, outputs, bindings, stream
            
    def __call__(self, data:np.ndarray):
        #多进程需要的
        self.cfx.push()

        np.copyto(self.inputs.host,data.ravel())
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.inputs.device, self.inputs.host, self.stream)
        # Run inference.
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.outputs.host,self.outputs.device, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        
        #多进程需要的
        self.cfx.pop()
        # Return only the host outputs.
        return self.outputs.host


if __name__ == "__main__":

    person_engine_path = '/home/xujiahong/trt_engine/fast-reid-fsid/fsid-res50-INT8.engine'
    model = FeatureExtractor(person_engine_path)

    name ='/home/xujiahong/NX/ReIDFsid_multithread_v2/persons.jpg'
    img = cv2.imread(name)

    def reid_thread(ImgEX, name):
        img = preprocessing_person(name)
        res = ImgEX(img)
        print(res[:5])
        print("now: {}".format(datetime.datetime.now().strftime('%H:%M:%S')))
        time.sleep(3)

    for i in range(5):
        t = threading.Thread(target=reid_thread, args=(model, name))
        t.start()






