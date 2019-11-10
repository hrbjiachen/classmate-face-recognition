from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
from create_classifier import training

datadir = './data/train_img'
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
print("Training Start")
obj = training(datadir, modeldir, classifier_filename)
model_path = obj.main_train()
print('Saved classifier model to file "%s"' % model_path)
sys.exit("All Done")
