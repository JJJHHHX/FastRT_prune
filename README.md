# C++ FastReID-TensorRT-pruned
convert structred pruned reid model to TensorRT engine, based on [fastreid/FastRT](https://github.com/JDAI-CV/fast-reid/tree/master/projects/FastRT)   
基于fastreid，将进行network-slimming后的模型转为TensorRT。[方法及实验链接](https://blog.csdn.net/qq_38853994/article/details/131781473?spm=1001.2014.3001.5501)
### How to Run

1. Generate '.wts' file  and pruned conv shape file from pytorch 

   ```
   python tools/get_wts.py --prune=True --pth-path="model_final.pth" \
   --wts-path="./model_final.wts" --shape-path="./pruned_conv_output_shape.txt"
   ```
2. change pruned conv shape file path in `fastrt/backbones/sbs_resnet.cpp` line 333.
   ```
   std::map<std::string, int> ConvOutputShapeMap=loadPrunedConvOutputShape("pruned_conv_output_shape_0.5.txt");
   ```

3. Build `fastrt` execute file
   
   ``` 
   mkdir build
   cd build
   cmake -DBUILD_FASTRT_ENGINE=ON \
         -DBUILD_DEMO=ON \
         -DUSE_CNUMPY=ON ..
   make
   ```

4. Run `fastrt`
   

   ``` 
   ./demo/fastrt -s  // serialize model & save as 'xxx.engine' file
   ./demo/fastrt -d  // deserialize 'xxx.engine' file and run inference
   ```

    
### <a name="ConfigSection"></a>`Tensorrt Model Config`

Edit `FastRT/demo/inference.cpp`, according to your model config


+ Ex1. `sbs_R50-ibn`
```
static const std::string WEIGHTS_PATH = "../sbs_R50-ibn.wts"; 
static const std::string ENGINE_PATH = "./sbs_R50-ibn.engine";

static const int MAX_BATCH_SIZE = 4;
static const int INPUT_H = 384;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 2048;
static const int DEVICE_ID = 0;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r50; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = true; 
static const bool WITH_NL = true;
static const int EMBEDDING_DIM = 0; 
```

+ Ex2. `sbs_R50_prune` with normalized output features
```
static const std::string WEIGHTS_PATH = "/pruned_0.5.wts"; 
static const std::string ENGINE_PATH = "./prune.engine";

static const int MAX_BATCH_SIZE = 1;
static const int INPUT_H = 256;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 2048;
static const int DEVICE_ID = 0;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r50_pruned; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::avgpool;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = false; 
static const bool WITH_NL = false; 
static const int EMBEDDING_DIM = 0; 
static const bool WITH_NORM = true;
```

### Supported conversion

*  Backbone: resnet50, resnet50-prune
*  Heads: embedding_head
*  Plugin layers: ibn, non-local
*  Pooling layers: maxpool, avgpool, GeneralizedMeanPooling, GeneralizedMeanPoolingP

