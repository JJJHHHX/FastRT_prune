import cv2
import numpy as np
from threading import Thread
from ReIDCpp.inference_clib import ReidEngine
import queue
import json
import time

NET_INPUT_SIZE_PERSON = (128, 256) # width, height
class AlgAppAb():
    def __init__(self, alg_name, batch_size: int = 1):
        self.name = alg_name
        self.batch_size = batch_size
        self.imgs = []
        self.msgs = []
        self.det_queue = queue.Queue()
        self.alg_res_queue = queue.Queue()
        self._initAlg()
        self._initThread()

    def __del__(self):
        self._uninitAlg()
    def _initAlg():
        raise NotImplementedError()
    def _uninitAlg():
        raise NotImplementedError()
    def handle(self, img_path:str,):
        self.imgs.append(img_path)
        if len(self.imgs) != self.batch_size:
            return None
        else:
            # alg_res = self._process(self.imgs, self.msgs)
            # self.imgs = []
            # self.msgs = []
            # return self._serializeResult(alg_res)
            self.det_queue.put(self.imgs)
            self.imgs = []
            self.msgs = []

class AlgApp(AlgAppAb):

    def _initAlg(self):
        print('Init extractor...')
        self.person_et = ReidEngine("/home/xujiahong/trt_engine/fast-reid-fsid/fsid-res50-INT8.engine")
        self.queue1 = queue.Queue()
        self.queue2 = queue.Queue()
        self.jress = []

    def _initThread(self):
        # 初始化线程
        print('Init threads...')
        push_thread = Thread(target=self._push_img,)
        push_thread.start()
        reid_thread = Thread(target=self._extract_feature,)
        reid_thread.start()

    def _uninitAlg(self):
        del self.person_et

    def _push_img(self):
        while True:
            if self.det_queue.empty() or self.queue1.qsize()>5:
                continue
            else:
                name = self.det_queue.get()
                img = cv2.imread(name[0])
                self.queue1.put(img)

    def _extract_feature(self):
        while True:
            if self.queue1.empty():
                continue
            else: 
                print("extracting...")
                img_patch= self.queue1.get()
                img_patch = self._exfeat_preprocess(img_patch, NET_INPUT_SIZE_PERSON)
                feat = self.person_et.run_reid(img_patch)
                self.jress.append(str(list(feat[:5])))
                if len(self.jress) == 3: 
                    result = {'alg_result': [self.jress]}
                    # send_data = json.dump(result)
                    with open('thread_cpp_sample.json', 'w') as f:
                        json.dump(result, f)

    def _exfeat_preprocess(self, img, resize):
        img = cv2.resize(img, resize)
        img = np.rollaxis(img, 2,0)
        ##### need BGR CHW
        return img


if __name__ == '__main__':
    alg_app = AlgApp("reid_cpp", 1)
    # img1= cv2.imread('/home/xujiahong/NX/ReIDFsid_bkVehicle_app_3.0.0/street.jpg')
    img= '/home/xujiahong/NX/ReIDFsid_multithread_v2/persons.jpg'
    start = time.time()
    for i in range(3):
        alg_app.handle(img)
    print("total time: {}s".format(time.time()-start))

    