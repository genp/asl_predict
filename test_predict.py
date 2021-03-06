#!/usr/bin/env python
import os, sys
import numpy as np
from asl_predict import Video, CNN, flatten
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import binarize
from sklearn.metrics import average_precision_score, accuracy_score

vid_feats_file = 'data/features/miohands_meanpool_10_flat_crop_center_rotate_5_flip_v.jbl'
if os.path.exists(vid_feats_file):
    vid_feats_dict = joblib.load(vid_feats_file)

else:
    feature = CNN()
    #DeepHands
    feature.set_params(model_def='models/1miohands-modelzoo-v2/test.prototxt', 
                       model_weights='models/1miohands-modelzoo-v2/1miohands-v2.caffemodel',
                       layer_name='loss3/SLclassifier')
    #VGG
    # feature.set_params(model_def='/Users/gen/kaizen/caffemodels/VGG/train.prototxt', 
    #                    model_weights='/Users/gen/kaizen/caffemodels/VGG/weights.caffemodel', 
    #                    layer_name='fc7')

    # frame_feats = v.extract_frame_features(feature)
    # pool_feats = v.extract_frame_features(feature, mean_pool_length=10)
    # flat_feat = flatten(pool_feats)

    # TODO: loading the lstm is not working
    # lstm = CNN()
    # lstm.set_params(model_def='/Users/gen/asl_predict/models/naacl15_vgg/poolmean.prototxt', 
    #                    model_weights='/Users/gen/asl_predict/models/naacl15_vgg/naacl15_pool_vgg_fc7_mean_fac2.caffemodel', 
    #                    layer_name='lstm2')

    # Load all features
    pool_length = 10
    vid_dir = 'data/videos/mp4'
    vid_feats = {}
    for fname in os.listdir(vid_dir):
        v = Video(os.path.join(vid_dir,fname))
        croptf = ('crop', [v.h/4, v.w/4, v.h/2, v.w/2])
        alltf = str((croptf, ('rotate', 5), ('flip_v', True)))
        v.get_frames(croptf, ('rotate', 5), ('flip_v', True))

        # Print some example frames
        if fname == os.listdir(vid_dir)[0]:
            print fname
            write_dir = 'data/frames'
            v.write_frames(write_dir, str(croptf))
            v.write_frames(write_dir, alltf)

        pool_feats = v.extract_frame_features(feature, str(croptf),mean_pool_length=pool_length)
        if str(croptf) not in vid_feats.keys():
            vid_feats[str(croptf)] = []
        vid_feats[str(croptf)].append(flatten(pool_feats))
        print 'Last feature shape {}'.format(vid_feats[str(croptf)][-1].shape)
        pool_feats = v.extract_frame_features(feature, alltf,mean_pool_length=pool_length)
        if alltf not in vid_feats.keys():
            vid_feats[alltf] = []
        vid_feats[alltf].append(flatten(pool_feats))
        print 'Last feature shape {}'.format(vid_feats[alltf][-1].shape)

    # Save all features
    vid_feats_dict = {}
    for key in vid_feats.keys():
        vid_feats_dict[key] = dict(zip(os.listdir(vid_dir), vid_feats[key]))
    joblib.dump(vid_feats_dict, vid_feats_file, compress=6)

# Read in labels
student_file = 'data/datamatrixFull.tsv'
student_lbls = {}
student_cats = []
exp_file = 'data/expertQueries.txt'
exp_lbls = {}
exp_cats = []

def read_lbls(fname, lbls):
    with open(fname, 'r') as f:
        headers = f.readline().split('\t')[2:]
        for row in f:
            items = row.split('\t')
            if items[0] not in lbls.keys():
                lbls[items[0]] = np.zeros((len(items)-2),)
            try:
                lbls[items[0]] += np.array([int(i) for i in items[2:]])
            except:
                print 'Error reading this line: '
                print items
                return   
        return headers

student_cats = read_lbls(student_file, student_lbls)    
exp_cats = read_lbls(exp_file, exp_lbls)    


def reduce(codes, ops, output_dim=100, alpha = 2.5):
    '''
    "codes" should be a numpy array of codes for either a single or multiple images of shape:
    (N, c) where "N" is the number of images and "c" is the length of codes.  

    "ops" indicates the processes to perform on the given feature.
    Currently supported operations: subsample, normalization (normalize), power normalization (power_norm)

    "output_dim" is the number of dimensions requested for output of a dimensionality reduction operation.
    Not needed for non dimensionality reduction operations (ie "normalization")
    
    "alpha" is the power for the power normalization operation
    '''
    output_codes = codes if len(codes.shape) > 1 else codes.reshape(1,len(codes))

    for op in ops:

        if op == "subsample":
            odim = output_dim
            if odim <= output_codes.shape[1]:
                output_codes = output_codes[:,0:odim]
            else:
                raise ValueError('output_dim is larger than the codes! ')
        elif op == "normalize":
            mean = np.mean(output_codes, 1)
            std = np.std(output_codes, 1)
            norm = np.divide((output_codes - mean[:, np.newaxis]),std[:, np.newaxis])
            output_codes = norm

        elif op == "power_norm":            
            pownorm = lambda x: np.power(np.abs(x), alpha)
            pw = pownorm(output_codes)
            norm = np.linalg.norm(pw, axis=1)
            if not np.any(norm):
                warnings.warn("Power norm not evaluated due to 0 value norm")
                continue
            output_codes = np.divide(pw,norm[:, np.newaxis])
            output_codes = np.nan_to_num(output_codes)

    if output_codes.shape[0] == 1:
        output_codes = np.reshape(output_codes, -1)
    return output_codes

slice = []
odim = vid_feats[alltf][-1].shape[0]
rdim = odim
for c in range(10):
    slice = slice + range(c*odim,c*odim+rdim)
student_attrs, student_Y_all = zip(*sorted(student_lbls.items(), key=lambda s: s[0].lower()))
inst_names = {}
inst_names['student'] = []
student_Y = []
student_feats = []
for idx, attr in enumerate(student_attrs):
    for key in vid_feats_dict.keys():
        inst_names['student'].append(attr)
        student_feats.append(vid_feats_dict[key][attr.replace(' ', '_')+'.mp4'][slice])
        student_Y.append(student_Y_all[idx])
student_Y = np.array(student_Y)
student_feats = reduce(np.array(student_feats), ['power_norm'], alpha=1)
student_Y = reduce(student_Y, ['normalize'])#np.divide(student_Y, np.max(student_Y))
print 'Student dataset: Y {} X {}'.format(student_Y.shape, student_feats.shape)

exp_attrs, exp_Y_all = zip(*sorted(exp_lbls.items(), key=lambda s: s[0].lower()))
exp_Y = []
exp_feats = []
inst_names['exp'] = []
for idx, attr in enumerate(exp_attrs):
    for key in vid_feats_dict.keys():
        inst_names['exp'].append(attr)
        exp_feats.append(vid_feats_dict[key][attr.replace(' ', '_')+'.mp4'][slice])
        exp_Y.append(exp_Y_all[idx])
exp_Y = np.array(exp_Y)
exp_feats = reduce(np.array(exp_feats), ['power_norm'], alpha=1)#np.array(exp_feats)
exp_Y = reduce(exp_Y, ['normalize'])#np.divide(exp_Y, np.max(exp_Y))
print 'Exp dataset: Y {} X {}'.format(exp_Y.shape, exp_feats.shape)

# Fit train - 1vs.Rest of concattenated mean pooled features
num_train = 70
binary_thresh = 0.2
X = np.array(student_feats[:num_train])
Y = binarize(student_Y[:num_train, :], threshold=binary_thresh)

classif = OneVsRestClassifier(SVC(kernel='linear',probability = True))
classif.fit(X, Y)

# This score function is not working for some reason, always returns 0
# score = classif.score(X, Y)
p = classif.predict(X)
scores = [accuracy_score(Y[:,i], p[:,i]) for i in range(Y.shape[1])]
ap = [average_precision_score(Y[:,i], p[:,i]) for i in range(Y.shape[1])]
chance = np.mean(np.divide(np.sum(Y,axis=0),Y.shape[1]))
print 'Student Train: Accuracy {} AP {} chance {}'.format(np.mean(scores), 
                                                          np.mean([s for s in ap if not np.isnan(s)]),
                                                          chance)

# Test
Yt = binarize(student_Y[num_train:, :], threshold=binary_thresh)
Xt = np.array(student_feats[num_train:])

p = classif.predict(Xt)
scores = [accuracy_score(Yt[:,i], p[:,i]) for i in range(Yt.shape[1])]
ap = [average_precision_score(Yt[:,i], p[:,i]) for i in range(Yt.shape[1])]
chance = np.mean(np.divide(np.sum(Yt,axis=0),Yt.shape[1]))
print 'Student Test: Accuracy {} AP {} chance {}'.format(np.mean(scores), 
                                                          np.mean([s for s in ap if not np.isnan(s)]),
                                                          chance)
print '*** Predicted Labels ***'
for idx, pred in enumerate(p):
    print '{} : {}'.format(inst_names['student'][num_train+idx], [student_cats[jdx] for jdx in range(len(student_cats)) if pred[jdx]])

# Fit train - 1vs.Rest of concattenated mean pooled features
X = np.array(exp_feats[:num_train])
Y = binarize(exp_Y[:num_train, :], threshold=binary_thresh)

classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X, Y)

p = classif.predict(X)
scores = [accuracy_score(Y[:,i], p[:,i]) for i in range(Y.shape[1])]
ap = [average_precision_score(Y[:,i], p[:,i]) for i in range(Y.shape[1])]
chance = np.mean(np.divide(np.sum(Y,axis=0),Y.shape[1]))
print 'Exp Train: Accuracy {} AP {} chance {}'.format(np.mean(scores), 
                                                          np.mean([s for s in ap if not np.isnan(s)]),
                                                          chance)

# Test
Yt = binarize(exp_Y[num_train:, :], threshold=binary_thresh)
Xt = np.array(exp_feats[num_train:])

p = classif.predict(Xt)
scores = [accuracy_score(Yt[:,i], p[:,i]) for i in range(Yt.shape[1])]
ap = [average_precision_score(Yt[:,i], p[:,i]) for i in range(Yt.shape[1])]
chance = np.mean(np.divide(np.sum(Yt,axis=0),Yt.shape[1]))
print 'Exp Test: Accuracy {} AP {} chance {}'.format(np.mean(scores), 
                                                          np.mean([s for s in ap if not np.isnan(s)]),
                                                          chance)
print '*** Predicted Labels ***'
for idx, pred in enumerate(p):
    print '{} : {}'.format(inst_names['student'][num_train+idx], [exp_cats[jdx] for jdx in range(len(exp_cats)) if pred[jdx]])
