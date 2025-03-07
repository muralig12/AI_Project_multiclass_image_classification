from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import bcrypt
import numpy
import tensorflow as tf
import requests
import subprocess
import json
from keras.engine import Model
from keras import models
from keras import layers
from keras.layers import Input
from keras import backend as K
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
import numpy as np
from keras_vggface import utils
import scipy as sp
import cv2
import time
import os
import glob
import pickle

app = Flask(__name__)
api = Api(app)



def pickle_stuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


def load_stuff(filename):
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


class FaceExtractor(object):
    """
    Singleton class to extraction face images from video files
    """
    CASE_PATH = "./pretrained_models/haarcascade_frontalface_alt.xml"

    def __new__(cls, weight_file=None, face_size=224):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceExtractor, cls).__new__(cls)
        return cls.instance

    def __init__(self, face_size=224):
        self.face_size = face_size

    def crop_face(self, imgarray, section, margin=20, size=224):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def extract_faces(self, video_file, save_folder):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        cap = cv2.VideoCapture(video_file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        print("length: {}, w x h: {} x {}, fps: {}".format(length, width, height, fps))
        # infinite loop, break by key ESC
        frame_counter = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame_counter = frame_counter + 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(64, 64)
                )
                # only keep the biggest face as the main subject
                face = None
                if len(faces) > 1:  # Get the largest face as main face
                    face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # area = w * h
                elif len(faces) == 1:
                    face = faces[0]
                if face is not None:
                    face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                    (x, y, w, h) = cropped
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    cv2.imshow('Faces', frame)
                    imgfile = os.path.basename(video_file).replace(".","_") +"_"+ str(frame_counter) + ".png"
                    imgfile = os.path.join(save_folder, imgfile)
                    cv2.imwrite(imgfile, face_img)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()



       



class Classify(Resource):
    def post(self):
        resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                                    pooling='avg')  # pooling: None, avg or max

        def image2x(image_path):
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=1)  # or version=2
            return x

        def cal_mean_feature(image_folder):
            face_images = list(glob.iglob(os.path.join(image_folder, '*.*')))

            def chunks(l, n):
                """Yield successive n-sized chunks from l."""
                for i in range(0, len(l), n):
                    yield l[i:i + n]

            batch_size = 32
            face_images_chunks = chunks(face_images, batch_size)
            fvecs = None
            for face_images_chunk in face_images_chunks:
                images = np.concatenate([image2x(face_image) for face_image in face_images_chunk])
                batch_fvecs = resnet50_features.predict(images)
                if fvecs is None:
                    fvecs = batch_fvecs
                else:
                    fvecs = np.append(fvecs, batch_fvecs, axis=0)
            return np.array(fvecs).sum(axis=0) / len(fvecs)

        FACE_IMAGES_FOLDER = "./data/face_images"
        VIDEOS_FOLDER = "./data/videos"
        extractor = FaceExtractor()
        folders = list(glob.iglob(os.path.join(VIDEOS_FOLDER, '*')))
        os.makedirs(FACE_IMAGES_FOLDER)
        names = [os.path.basename(folder) for folder in folders]
        for i, folder in enumerate(folders):
            name = names[i]
            videos = list(glob.iglob(os.path.join(folder, '*.*')))
            save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
            print(save_folder)
            os.makedirs(save_folder)
            for video in videos:
                extractor.extract_faces(video, save_folder)

        precompute_features = []
        for i, folder in enumerate(folders):
            name = names[i]
            save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
            mean_features = cal_mean_feature(image_folder=save_folder)
            precompute_features.append({"name": name, "features": mean_features})
        pickle_stuff("./data/precompute_features.pickle", precompute_features)

        retJson = {
            "status": 200,
            "msg": "MODEL successfully CREATED"
        }
        return jsonify(retJson)




class FaceIdentify(object):
    """
    Singleton class for real time face identification
    """
    CASE_PATH = "./pretrained_models/haarcascade_frontalface_alt.xml"

    def __new__(cls, precompute_features_file=None):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceIdentify, cls).__new__(cls)
        return cls.instance

    def __init__(self, precompute_features_file=None):
        self.face_size = 224
        self.precompute_features_map = load_stuff(precompute_features_file)
        print("Loading VGG Face model...")
        self.model = VGGFace(model='resnet50',
                             include_top=False,
                             input_shape=(224, 224, 3),
                             pooling='avg')  # pooling: None, avg or max
        print("Loading VGG Face model done")

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=20, size=224):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def identify_face(self, features, threshold=100):
        distances = []
        for person in self.precompute_features_map:
            person_features = person.get("features")
            distance = sp.spatial.distance.euclidean(person_features, features)
            distances.append(distance)
        min_distance_value = min(distances)
        min_distance_index = distances.index(min_distance_value)
        if min_distance_value < threshold:
            return self.precompute_features_map[min_distance_index].get("name")
        else:
            return "?"

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture("./data/test/video1.mp4")
        # infinite loop, break by key ESC
        while True:
            if not video_capture.isOpened():
                time.sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(64, 64)
            )
            # placeholder for cropped faces
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=10, size=self.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                face_imgs[i, :, :, :] = face_img
            if len(face_imgs) > 0:
                # generate features for each face
                features_faces = self.model.predict(face_imgs)
                predicted_names = [self.identify_face(features_face) for features_face in features_faces]
            # draw results
            for i, face in enumerate(faces):
                label = "{}".format(predicted_names[i])
                self.draw_label(frame, (face[0], face[1]), label)

            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


class Predict(Resource):
    def post(self):
        image_path = request.get_json()
        face = FaceIdentify(precompute_features_file="./data/precompute_features.pickle")
        #K.clear_session()
        face.detect_face()

        #return jsonify(retJson)


api.add_resource(Classify, '/classify')
api.add_resource(Predict, '/predict')

if __name__=="__main__":
    app.run(debug=True,port=8008)
