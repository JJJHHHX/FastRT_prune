import tensorrt as trt
# trt.init_libnvinfer_plugins(None, "")
import cv2
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda

import time

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()

def preprocessing(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)
    image = image[:,:,::-1]
    # image = np.full((256,128,3), 255, dtype=np.uint8)
    # need CHW (RGB)
    return np.rollaxis(image, 2,0)
   
class GetFeature:
    
    def __init__(self,engine_path,max_batch_size=1):
        
        self.engine_path = engine_path
        self.batch_size = max_batch_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()
            
    @staticmethod
    def load_engine(trt_runtime, engine_path):           
        with open(engine_path, 'rb') as f, trt_runtime:
            return trt_runtime.deserialize_cuda_engine(f.read())
    
    def allocate_buffers(self):
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))*self.engine.max_batch_size
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
    def infer_curr_batch(self):
        pass
            
    # def __call__(self, data:np.ndarray):

    #     np.copyto(self.inputs.host[:data.ravel().shape[0]],data.ravel())
    #     # Transfer input data to the GPU.
    #     t1 = time.perf_counter()
    #     cuda.memcpy_htod_async(self.inputs.device, self.inputs.host, self.stream)
    #     # Run inference.
    #     t2 = time.perf_counter()
    #     self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
    #     # Transfer predictions back from the GPU.
    #     t3 = time.perf_counter()
    #     cuda.memcpy_dtoh_async(self.outputs.host,self.outputs.device, self.stream)
    #     # Synchronize the stream
    #     t4 = time.perf_counter()
    #     self.stream.synchronize()
    #     t5 = time.perf_counter()
    #     # Return only the host outputs.
    #     print("{:<30}:{:.3f} ms".format("host to device time",(t2-t1)*1000))
    #     print("{:<30}:{:.3f} ms".format("gpu inference time",(t3-t2)*1000))
    #     print("{:<30}:{:.3f} ms".format("device to host time",(t4-t3)*1000))
    #     print("{:<30}:{:.3f} ms".format("systime inference time",(t5-t1)*1000))
    #     print("================")
    #     return self.outputs.host

    '''数据传输和gpu计算同步'''
    def __call__(self, data:np.ndarray):

        np.copyto(self.inputs.host[:data.ravel().shape[0]],data.ravel())
        # Transfer input data to the GPU.
        t1 = time.perf_counter()
        cuda.memcpy_htod(self.inputs.device, self.inputs.host)
        # Run inference.
        t2 = time.perf_counter()
        self.context.execute(batch_size=self.batch_size, bindings=self.bindings)
        # Transfer predictions back from the GPU.
        t3 = time.perf_counter()
        cuda.memcpy_dtoh(self.outputs.host,self.outputs.device)
        # Synchronize the stream
        t4 = time.perf_counter()
        # Return only the host outputs.
        print("{:<30}:{:.3f} ms".format("host to device time",(t2-t1)*1000))
        print("{:<30}:{:.3f} ms".format("gpu inference time",(t3-t2)*1000))
        print("{:<30}:{:.3f} ms".format("device to host time",(t4-t3)*1000))
        print("================")
        return self.outputs.host

if __name__ == "__main__":
    
    # trt_engine_path = '/home/xujiahong/trt_engine/fast-reid-fsid/fsid-res50.engine'
    # trt_engine_path = '/home/xujiahong/trt_engine/FastRT/build/fsid-res50-test.engine'
    # trt_engine_path = '/home/xujiahong/trt_engine/fast-reid-fsid/outputs/trt_model/baseline.engine'
    trt_engine_path = '/home/xujiahong/trt_engine/FastRT/fsid-res50-INT8-bs8.engine'
    model = GetFeature(trt_engine_path)
    # img = preprocessing('/home/xujiahong/NX/ReIDFsid_multithread_v2/r.jpg')

        
    names = ['/home/xujiahong/PCL_benchmark/img_person/Market-1501-v15.09.15/bounding_box_test/1216_c3s3_013728_03.jpg',
    '/home/xujiahong/PCL_benchmark/img_person/Market-1501-v15.09.15/bounding_box_test/1499_c6s3_088642_02.jpg',
    '/home/xujiahong/PCL_benchmark/img_person/Market-1501-v15.09.15/bounding_box_test/0310_c1s2_004041_01.jpg']
    # img = preprocessing('/home/xujiahong/PCL_benchmark/img_person/Market-1501-v15.09.15/bounding_box_test/1216_c3s3_013728_03.jpg')
    from tqdm import tqdm
    import time
    start = time.time()
    for i in range(len(names)):
        img = preprocessing(names[i])
        res = model(img)
        print(res[:10])
    # print(time.time() - start)
    # print(res[:10])
