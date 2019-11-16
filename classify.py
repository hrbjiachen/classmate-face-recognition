import tensorflow as tf
import numpy as np
import os
import detect_face
import facenet
import pickle
import cv2
from scipy import misc
from preprocess import PreProcessor
pp = PreProcessor()
class Classify():
    def __init__(self, modeldir, classifier_filename, train_img="./data/train_img"):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, log_device_placement=False))
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(
                    self.sess, './npy')

            self.minsize = 20  # minimum size of face
            self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            self.factor = 0.709  # scale factor
            self.image_size = 160
            self.margin = 44
            self.frame_interval = 3
            self.HumanNames = os.listdir(train_img)
            self.HumanNames.sort()

            # load model
            facenet.load_model(modeldir)
            with open(os.path.expanduser(classifier_filename), 'rb') as infile:
                (self.model, self.class_names) = pickle.load(infile)

            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph(
            ).get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]

    def __del__(self):
        self.sess.close()

    def classify_image(self, img_path):
        img_path = os.path.expanduser(img_path)

        # read image
        frame = cv2.imread(img_path)

        # resize frame (optional)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        bb_array, text_array = self.predict(frame)
        self.show_result_img(frame, bb_array, text_array)

        cv2.destroyAllWindows()

    def classify_video(self, video_path):
        print('Start Recognition')
        video_capture = cv2.VideoCapture(video_path)

        while True:
            ret, frame = video_capture.read()

            if frame is None:
                break

            bb_array, text_array = self.predict(frame)
            stop = self.show_result_img(frame, bb_array, text_array, True)
            if stop:
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def classify_webcam(self):
        print('Start Recognition')
        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()
            if frame is None:
                break
            bb_array, text_array = self.predict(frame)
            stop = self.show_result_img(frame, bb_array, text_array, True)
            if stop:
                break

        camera.release()
        cv2.destroyAllWindows()

    # Model prediction, return face name, position for plotting
    def predict(self, frame):
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]
        bounding_boxes, _ = self.detect_face(frame)

        num_faces = bounding_boxes.shape[0]
        print('Detected_FaceNum: %d' % num_faces)

        bb_array = []
        text_array = []

        if num_faces > 0:
            for i in range(num_faces):
                bb, text = self.facenet_predict_proba(frame, bounding_boxes, i)
                bb_array.append(bb)
                text_array.append(text)
        else:
            print('Unable to identify face')
        return bb_array, text_array

    def facenet_predict_proba(self, frame, bounding_boxes, face_index):
        text = []
        emb_array, bb = self.getImageEmbeddings(frame, bounding_boxes, face_index)

        predictions = self.model.predict_proba(emb_array)

        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(
            len(best_class_indices)), best_class_indices]

        print(predictions)
        print(best_class_probabilities)

        # calculate position to plot result under box
        text_x = bb[0]
        text_y = bb[3] + 20
        print('Result Indices: ', best_class_indices[0])

        HumanNames = self.HumanNames
        print(HumanNames)

        if best_class_probabilities > 0.5:
            result_names = HumanNames[best_class_indices[0]]
            text = [result_names, text_x, text_y, best_class_probabilities]
        return bb, text

    def show_result_img(self, frame, bb_array, text_array, video=False):
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        frame = frame[:, :, 0:3]

        for bb in bb_array:
            cv2.rectangle(frame, (bb[0], bb[1]),
                          (bb[2], bb[3]), (0, 255, 0), 2)

        for text in text_array:
            if len(text) > 0:
                prob = np.array2string(text[3], formatter={
                                       'float_kind': lambda x: "%.2f" % x})
                cv2.putText(frame, text[0], (text[1], text[2]),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
                cv2.putText(frame, prob, (text[1], text[2]-25),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1, lineType=2)
        # show
        if video:
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True
        else:
            cv2.imshow('image', frame)
            cv2.waitKey(0)

        return False

    def detect_face(self, img):
        return detect_face.detect_face(
            img,
            self.minsize,
            self.pnet,
            self.rnet,
            self.onet,
            self.threshold,
            self.factor
        )

    def getImageEmbeddings(self, frame, bounding_boxes, face_index):
        emb_array = np.zeros((1, self.embedding_size))

        scaled, bb = self.getCroppedImage(frame, bounding_boxes, face_index)

        scaled = cv2.resize(scaled, (self.image_size, self.image_size),
                            interpolation=cv2.INTER_CUBIC)
        scaled = facenet.prewhiten(scaled)
        scaled_reshape = scaled.reshape(-1,
                                        self.image_size, self.image_size, 3)
        feed_dict = {
            self.images_placeholder: scaled_reshape, self.phase_train_placeholder: False}
        emb_array[0, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)

        return emb_array, bb

    def getCroppedImage(self, frame, bounding_boxes, face_index):
        bb = np.zeros(4, dtype=np.int32)
        det = bounding_boxes[:, 0:4]

        bb[0] = det[face_index][0] if det[face_index][0] > 0 else 0
        bb[1] = det[face_index][1] if det[face_index][1] > 0 else 0
        bb[2] = det[face_index][2] if det[face_index][2] < len(
            frame[0]) else len(frame[0])
        bb[3] = det[face_index][3] if det[face_index][3] < len(
            frame) else len(frame)

        """ if bb[0] <= 0 or bb[1] <= 0 or bb[2] >= len(frame[0]) or bb[3] >= len(frame):
            print('face is too close') """

        cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
        cropped = facenet.flip(cropped, False)
        scaled = misc.imresize(
            cropped, (self.image_size, self.image_size), interp='bilinear')
        return scaled, bb
