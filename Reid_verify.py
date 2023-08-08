import tensorrt as trt
# import required modules
import cv2
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from PIL import Image

TRT_LOGGER = trt.Logger()

# Filenames of TensorRT plan file and input/output images.
engine_file = "./build/fsid-res50.engine"

# utils for input processing 
def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 256), interpolation=cv2.INTER_CUBIC)

    # image = image[:,:,::-1]
    # image = np.full((256, 128, 3), 255, dtype=np.uint8)
    # Mean normalization
    mean = np.array([123.675, 116.28, 103.53]).astype('float32')
    stddev = np.array([58.395, 57.120000000000005, 57.375]).astype('float32')
    data = (np.asarray(image).astype('float32')  - mean) / stddev
    # Switch from HWC to to CHW order
    data = np.rollaxis(data,2,0)
    data = np.concatenate(np.reshape(data[:,:,::-1], (3, 256*128)))
    return data
    # return np.rollaxis(data,2,0)

# load tensorRT engine
# def load_engine(engine_file_path):
#     assert os.path.exists(engine_file_path)
#     print("Reading engine from file {}".format(engine_file_path))
#     with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#         return runtime.deserialize_cuda_engine(f.read())

def load_engine(engine_path):
    assert os.path.exists(engine_path)
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    trt_runtime = trt.Runtime(TRT_LOGGER)
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

#### inference pipeline
def infer(engine, input_file):
    print("Reading input image from file {}".format(input_file))
    input_image = preprocess(input_file)
    # input_image = np.expand_dims(input_image, axis=0)
    print('input shape', input_image.shape)
    # input_image = np.expand_dims(input_image, axis=0)

    with engine.create_execution_context() as context:
        # Set input shape based on image dimensions for inference
        # context.set_binding_shape(engine.get_binding_index("input"), ( image_height, image_width, 3))
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_buffer.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))

        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()
        print(output_buffer)

print("Running TensorRT inference for FCN-ResNet101")
# with load_engine(engine_file) as engine:
#     infer(engine, \
#     input_file='/home/xujiahong/PCL_benchmark/img_person/Market-1501-v15.09.15/bounding_box_test/1216_c3s3_013728_03.jpg')

engine= load_engine(engine_file)
infer(engine, \
    input_file='/home/xujiahong/PCL_benchmark/img_person/Market-1501-v15.09.15/bounding_box_test/1216_c3s3_013728_03.jpg')