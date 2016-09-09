#!/usr/bin/env python
import os
import numpy as np
from asl_predict import Video, CNN, flatten
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import binarize
from sklearn.metrics import average_precision_score, accuracy_score

# v = Video('data/videos/mp4/accident.mp4')
# v.get_frames()
feature = CNN()
feature.set_params(model_def='/Users/gen/kaizen/caffemodels/VGG/train.prototxt', 
                   model_weights='/Users/gen/kaizen/caffemodels/VGG/weights.caffemodel', 
                   layer_name='fc7')
# frame_feats = v.extract_frame_features(feature)
# pool_feats = v.extract_frame_features(feature, mean_pool_length=10)
# flat_feat = flatten(pool_feats)

# TODO: loading the lstm is not working
# lstm = CNN()
# lstm.set_params(model_def='/Users/gen/asl_predict/models/naacl15_vgg/poolmean.prototxt', 
#                    model_weights='/Users/gen/asl_predict/models/naacl15_vgg/naacl15_pool_vgg_fc7_mean_fac2.caffemodel', 
#                    layer_name='lstm2')

# Load all features
vid_feats_file = 'data/features/vgg_meanpool_5_flat.jbl'
if os.path.exists(vid_feats_file):
    vid_feats_dict = joblib.load(vid_feats_file)
    # vid_feats = zip(*sorted(vid_feats_dict.items(), key=lambda s: s[0].lower()))[1]
else:
    vid_dir = 'data/videos/mp4'
    vid_feats = []
    for fname in os.listdir(vid_dir):
        v = Video(os.path.join(vid_dir,fname))
        v.get_frames()
        pool_feats = v.extract_frame_features(feature, mean_pool_length=5)
        vid_feats.append(flatten(pool_feats))
        print 'Last feature shape {}'.format(vid_feats[-1].shape)

    # Save all features
    vid_feats_dict = dict(zip(os.listdir(vid_dir), vid_feats))
    joblib.dump(vid_feats, vid_feats_file, compress=6)

# Read in labels
student_file = 'data/datamatrixFull.tsv'
student_lbls = {}

exp_file = 'data/expertQueries.txt'
exp_lbls = {}

def read_lbls(fname, lbls):
    with open(fname, 'r') as f:
        _ = f.readline()
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

read_lbls(student_file, student_lbls)    
read_lbls(exp_file, exp_lbls)    

student_attrs, student_Y = zip(*sorted(student_lbls.items(), key=lambda s: s[0].lower()))
student_Y = np.array(student_Y)
student_feats = []
for attr in student_attrs:
    student_feats.append(vid_feats_dict[attr.replace(' ', '_')+'.mp4'])
student_feats = np.array(student_feats)
student_Y = np.divide(student_Y, np.max(student_Y))
print 'Student dataset: Y {} X {}'.format(student_Y.shape, student_feats.shape)

exp_attrs, exp_Y = zip(*sorted(exp_lbls.items(), key=lambda s: s[0].lower()))
exp_Y = np.array(exp_Y)
exp_feats = []
for attr in exp_attrs:
    exp_feats.append(vid_feats_dict[attr.replace(' ', '_')+'.mp4'])
exp_feats = np.array(exp_feats)
exp_Y = np.divide(exp_Y, np.max(exp_Y))
print 'Exp dataset: Y {} X {}'.format(exp_Y.shape, exp_feats.shape)

# Fit train - 1vs.Rest of concattenated mean pooled features
num_train = 50
X = np.array(student_feats[:num_train])
Y = binarize(student_Y[:num_train, :], threshold=0.1)

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
Yt = binarize(student_Y[num_train:, :], threshold=0.1)
Xt = np.array(student_feats[num_train:])

p = classif.predict(Xt)
scores = [accuracy_score(Yt[:,i], p[:,i]) for i in range(Yt.shape[1])]
ap = [average_precision_score(Yt[:,i], p[:,i]) for i in range(Yt.shape[1])]
chance = np.mean(np.divide(np.sum(Yt,axis=0),Yt.shape[1]))
print 'Student Train: Accuracy {} AP {} chance {}'.format(np.mean(scores), 
                                                          np.mean([s for s in ap if not np.isnan(s)]),
                                                          chance)


# Fit train - 1vs.Rest of concattenated mean pooled features
num_train = 50
X = np.array(exp_feats[:num_train])
Y = binarize(exp_Y[:num_train, :], threshold=0.1)

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
Yt = binarize(exp_Y[num_train:, :], threshold=0.1)
Xt = np.array(exp_feats[num_train:])

p = classif.predict(Xt)
scores = [accuracy_score(Yt[:,i], p[:,i]) for i in range(Yt.shape[1])]
ap = [average_precision_score(Yt[:,i], p[:,i]) for i in range(Yt.shape[1])]
chance = np.mean(np.divide(np.sum(Yt,axis=0),Yt.shape[1]))
print 'Exp Train: Accuracy {} AP {} chance {}'.format(np.mean(scores), 
                                                          np.mean([s for s in ap if not np.isnan(s)]),
                                                          chance)
