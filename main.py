from classify import Classify
import os

modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
img_path = './test/test4.jpg'
video_path = './test/video/test_video3.mp4'

cfy = Classify(modeldir, classifier_filename)

# classify image
# cfy.classify_image(img_path)

# classify video
# cfy.classify_video(video_path)

# classify webcam
cfy.classify_webcam()

