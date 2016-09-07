#!/usr/bin/env python
import numpy as np


class Video:
    def __init__(self):
        pass
    def get_frames(self):
        pass
    def pool(self):
        pass
    def predict(self):
        pass

class Frame:
    def __init__(self):
        pass
    def predict(self):
        pass


class CNN():

    MANY_BATCH_SIZE = 500
    CACHE = {}

    def get_networks(self):
        key = self.cache_key()
        if not key in CNN.CACHE.keys():
            self.populate_cache(key)
        self.single = CNN.CACHE[key].single
        self.many = CNN.CACHE[key].many

    def del_networks(self):
        CNN.CACHE = {}
        self.single = None
        self.many = None

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

        def_path = "caffemodels/" + self.model +"/train.prototxt"
        weight_path = "caffemodels/" + self.model + "/weights.caffemodel"
        #go through and edit batch size
        arch = open(def_path,'r').readlines()
        for i in range(len(arch)):
            if "batch_size" in arch[i]:
                arch[i] = re.sub('\d+',str(batch_size),arch[i])
        temp.writelines(arch)
        temp.close()

        net = caffe.Net(str(temp.name), str(weight_path), caffe.TEST)
        xform = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        xform.set_transpose('data', self.transpose)
        xform.set_channel_swap('data',self.channel_swap)
        
        # TODO delete temp file

        return CNN_Model(net, xform)


    def set_params(self, **kwargs):

        '''
        Parameters
        ------------
        "model" is the folder name where the model specs and weights live. 
        ie model = "VGG", "GoogleNet", "BVLC_Reference_Caffenet"
        
        "layer_name" is the layer name used for extraction 
        ie layer_name = "fc7" (for VGG)
        
        see below for better idea of what "transpose" and
        "channel_swap" are used for
        http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

        set "initialize" to False when using extract_many.  Initialize
        makes single-patch feature extraction significantly faster

        '''

        ReducibleFeature.set_params(self, **kwargs)        
        self.model = kwargs.get('model', "caffenet")
        self.layer_name = kwargs.get('layer_name', "fc7")
        self.transpose = kwargs.get('transpose', (2,0,1))
        self.channel_swap = kwargs.get('channel_swap', (2,1,0))


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
    
    def extract_many(self, imgs):
        '''
        imgs is a list of app.models.Patch.image, which are ndarrays of shape (x,y,3)
        '''
        self.get_networks()

        if len(imgs) > CNN.MANY_BATCH_SIZE:
            print 'exceeded max batch size. splitting into {} minibatches'.format(int(len(imgs)/CNN.MANY_BATCH_SIZE)+1)
            codes = np.asarray([])
            for i in range(int(len(imgs)/CNN.MANY_BATCH_SIZE)+1):
                tim = imgs[i*CNN.MANY_BATCH_SIZE:min(len(imgs),(i+1)*CNN.MANY_BATCH_SIZE)]
                tim = np.array([self.many.xform.preprocess('data',i) for i in tim])
                num_imgs = len(tim)
                if num_imgs < CNN.MANY_BATCH_SIZE:
                    tim = np.vstack((tim, np.zeros(np.append(CNN.MANY_BATCH_SIZE-num_imgs,self.many.net.blobs['data'].data.shape[1:]),dtype=np.float32)))                 
                self.many.net.set_input_arrays(tim, np.ones(CNN.MANY_BATCH_SIZE,dtype=np.float32))
                p = self.many.net.forward()
                codes = np.append(codes,self.many.net.blobs[self.layer_name].data[...])
            codes = codes.reshape(np.append(-1,self.many.net.blobs[self.layer_name].data.shape[1:]))
            codes = codes[:len(imgs), :]
        else:
            tim = np.array([self.many.xform.preprocess('data',i) for i in imgs])
            num_imgs = len(tim)
            if num_imgs < CNN.MANY_BATCH_SIZE:
                tim = np.vstack((tim, np.zeros(np.append(CNN.MANY_BATCH_SIZE-num_imgs,self.many.net.blobs['data'].data.shape[1:]),dtype=np.float32)))
            self.many.net.set_input_arrays(tim, np.ones(tim.shape[0],dtype=np.float32))
            p = self.many.net.forward()
            codes = self.many.net.blobs[self.layer_name].data[...]
            if num_imgs < CNN.MANY_BATCH_SIZE:
                codes = codes[:num_imgs,:]
        return codes
