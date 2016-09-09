#!/usr/bin/env python
import copy
import re
import tempfile

import cv2
import numpy as np


class Video():
    ''' This is the object that holds a video file and contains
    methods for processing that file and predicting its contents. 
    '''
    def __init__(self, fname):
        self.vidcap = cv2.VideoCapture(fname)
        self.fname = fname
        self.w = self.vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self.h = self.vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.fps = round(self.vidcap.get(cv2.cv.CV_CAP_PROP_FPS))
        self.num_frames = self.vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        self.frames = []
        print self
        
    def __repr__(self):
        return ("Video from {}: "
               "(Width, Height, FPS, Number of Frames) "
               "({}, {}, {}, {})").format(self.fname, 
                                          self.w, self.h, 
                                          self.fps, self.num_frames)

    def get_frames(self):
        if self.frames:
            return self.frames

        self.vidcap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, 0.0)
        count = 0
        success = 1
        while success:
          success,image = self.vidcap.read()
          if not success:
              break
          self.frames.append(Frame(image))
          count += 1
          if count % 5 == 0:
              print 'Read {} of {} frames'.format(count, 
                                                  self.num_frames)
        print 'Done reading {} frames'.format(self.num_frames)
        
    def write_frames(self, write_dir):
        if not self.frames:
            self.get_frames()
        for frame in self.frames:
            cv2.imwrite(os.path.join(write_dir, 
                                     'frame_{}.jpg'.format(count)), 
                        frame.image)

    def extract_frame_features(self, feature, mean_pool_length=0):
        frame_feats = [copy.deepcopy(x.extract_feature(feature)) for x in self.frames]
        if mean_pool_length > 0:
            frame_feats = mean_pool(frame_feats, mean_pool_length)
        return frame_feats

    def predict(self):
        pass

class Frame():
    def __init__(self, image):
        self.image = image

    def extract_feature(self, feature):
        return feature.extract(self.image)

class Estimator():
    def train(self):
        pass
    def predict(self):
        pass

class MultiNet:
   def __init__(self, single, many):
       self.single = single
       self.many = many

class CNN_Model:
   def __init__(self, net, xform):
       self.net = net
       self.xform = xform

class CNN():

    MANY_BATCH_SIZE = 500
    CACHE = {}

    def get_networks(self):
        key = self.cache_key()
        if not key in CNN.CACHE.keys():
            self.populate_cache(key)
        self.single = CNN.CACHE[key].single

    def del_networks(self):
        CNN.CACHE = {}
        self.single = None

    def cache_key(self):
        key = str(self.params)
        return key
        
    def populate_cache(self, key):
        single = self.create_model(1)
        many = self.create_model(CNN.MANY_BATCH_SIZE)
        CNN.CACHE[key] = MultiNet(single, many)

    def create_model(self, batch_size):
        print 'creating model'
        import caffe
        import config
        if config.USE_GPU:                
                caffe.set_device(config.GPU_DEVICE_ID)
                caffe.set_mode_gpu()
        temp = tempfile.NamedTemporaryFile(delete = False)

        #go through and edit batch size
        arch = open(self.model_def,'r').readlines()
        for i in range(len(arch)):
            if "batch_size" in arch[i]:
                arch[i] = re.sub('\d+',str(batch_size),arch[i])
        temp.writelines(arch)
        temp.close()

        net = caffe.Net(str(temp.name), caffe.TEST, weights=self.model_weights)
        xform = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        xform.set_transpose('data', self.transpose)
        xform.set_channel_swap('data',self.channel_swap)
        
        return CNN_Model(net, xform)


    def set_params(self, **kwargs):

        '''
        Parameters
        ------------
        "model_def" is the prototxt file name where the model is defined.
        "model_weights" is the caffemodel file with the weight values for the pre-trained model. 
        ie models such as "VGG", "BVLC_Reference_Caffenet"
        
        "layer_name" is the layer name used for extraction 
        ie layer_name = "fc7" (for VGG)
        
        see below for better idea of what "transpose" and
        "channel_swap" are used for
        http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

        set "initialize" to False when using extract_many.  Initialize
        makes single-patch feature extraction significantly faster

        '''

        self.model_def = kwargs.get('model_def', "")
        self.model_weights = kwargs.get('model_weights', "")
        self.layer_name = kwargs.get('layer_name', "fc7")
        self.transpose = kwargs.get('transpose', (2,0,1))
        self.channel_swap = kwargs.get('channel_swap', (2,1,0))
        self.params = kwargs    


    #assume that we're getting a single image
    #Img comes in format (x,y,c)
    def extract(self, img):
        # check that network is initialized
        self.get_networks()

        img = self.single.xform.preprocess('data',img)
        if len(img.shape) == 3:
            img = np.expand_dims(img,axis=0)
        self.single.net.set_input_arrays(img, np.array([1], dtype=np.float32))
        p = self.single.net.forward()
        feat = self.single.net.blobs[self.layer_name].data[...].reshape(-1)
        feat = np.reshape(feat, (-1))
        return feat


def mean_pool(feats, output_length):
    step = np.true_divide(len(feats),output_length)
    pool_feats = []
    cur_pos = 0.0
    while cur_pos < len(feats):
        # average the frames in this group
        pool_feats.append(np.mean(np.array(feats[int(cur_pos):min(len(feats), int(cur_pos+step))]), axis=0))
        cur_pos = cur_pos+step
    return pool_feats

def flatten(feats):
    return np.reshape(np.array(feats), (-1))
