from scipy import misc
import tensorflow as tf
import numpy as np
import detect_face
import facenet
import os


class PreProcessor():
    def __init__(self):
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

    def align(self, image_path, output_path):
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            print(errorMessage)
        else:
            cropped_image = self.crop_image(img)
            if len(cropped_image) != 0:
                print("Processing image: %s" % image_path)
                misc.imsave(output_path, cropped_image)
            else:
                print("Unable to align image - %s" % image_path)

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

    def crop_image(self, img):
        cropped = []
        scaled = []

        if img.ndim < 2:
            return
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:, :, 0:3]

        bounding_boxes, _ = self.detect_face(img)
        num_faces = bounding_boxes.shape[0]

        bb = np.zeros(4, dtype=np.int32)

        if num_faces > 0:
            img_size = np.asarray(img.shape)[0:2]
            det = self.get_det(bounding_boxes, img_size)
            bb[0] = np.maximum(det[0]-self.margin/2, 0)
            bb[1] = np.maximum(det[1]-self.margin/2, 0)
            bb[2] = np.minimum(det[2]+self.margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+self.margin/2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(
                cropped, (self.image_size, self.image_size), interp='bilinear')
        return scaled

    def get_det(self, bounding_boxes, img_size):
        num_faces = bounding_boxes.shape[0]
        det = bounding_boxes[:, 0:4]
        if num_faces > 1:
            bounding_box_size = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])
            img_center = img_size / 2
            offset_x = (det[:, 0]+det[:, 2])/2-img_center[1]
            offset_y = (det[:, 1]+det[:, 3])/2-img_center[0]
            offsets = np.vstack([offset_x, offset_y])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            # some extra weight on the centering
            index = np.argmax(bounding_box_size - offset_dist_squared*2.0)
            det = (det[index, :])

        return np.squeeze(det)

    # in_folder must use facenet.get_dataset
    def align_dir(self, in_folder, out_folder_dir):
        images_total = 0

        for image_path in in_folder.image_paths:
            images_total += 1

            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_file_path = os.path.join(
                out_folder_dir, filename + '.png')

            if not os.path.exists(output_file_path):
                self.align(image_path, output_file_path)
            else:
                print("Output file path already exist- %s" % output_file_path)

        return images_total
