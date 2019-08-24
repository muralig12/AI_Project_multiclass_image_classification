from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import bcrypt
import numpy
import tensorflow as tf
import requests
import subprocess
import json

import numpy as np
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import math
import cv2

app = Flask(__name__)
api = Api(app)




# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = './data1/model/bottleneck_fc_model.h6'

# number of epochs to train top model
epochs = 30
# batch size used by flow_from_directory and predict_generator
batch_size = 10


def save_bottlebeck_features(train_data_dir,validation_data_dir):
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))
    print(generator.classes)
    np.save('./data1/model/class_indices.npy', generator.class_indices)
    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples // batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('./data1/model/bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples // batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('./data1/model/bottleneck_features_validation.npy',
            bottleneck_features_validation)



       



class Classify(Resource):
    def post(self):
        postedData = request.get_json()
        train_data_dir = postedData["train_path"]
        validation_data_dir = postedData["valid_path"]
        #url = postedData["url"]


        save_bottlebeck_features(train_data_dir,validation_data_dir)
        datagen_top = ImageDataGenerator(rescale=1. / 255)
        generator_top = datagen_top.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

        nb_train_samples = len(generator_top.filenames)
        num_classes = len(generator_top.class_indices)

        # save the class indices to use use later in predictions
        np.save('./data1/model/class_indices.npy', generator_top.class_indices)

        # load the bottleneck features saved earlier
        train_data = np.load('./data1/model/bottleneck_features_train.npy')

        # get the class lebels for the training data, in the original order
        train_labels = generator_top.classes

        # https://github.com/fchollet/keras/issues/3467
        # convert the training labels to categorical vectors
        train_labels = to_categorical(train_labels, num_classes=num_classes)

        generator_top = datagen_top.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

        nb_validation_samples = len(generator_top.filenames)
        print(nb_validation_samples)

        validation_data = np.load('./data1/model/bottleneck_features_validation.npy')

        validation_samples=generator_top.samples
        print("validation_samples=",validation_samples)
        validation_labels = generator_top.classes
        validation_labels = to_categorical(
            validation_labels, num_classes=num_classes)

        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        print(validation_labels)

        model.fit(train_data, train_labels,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(validation_data, validation_labels))

        model.save_weights(top_model_weights_path)

        (eval_loss, eval_accuracy) = model.evaluate(
            validation_data, validation_labels, batch_size=batch_size, verbose=1)
        print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
        print("[INFO] Loss: {}".format(eval_loss))
        """

        response = requests.get("http://127.0.0.1:5000")
        response.json()
        #we got the response for the '/' request.

        response = requests.get("http://127.0.0.1:5000/quarks")
        response.json()

        #We can convert the response to JSON object and play with it:
        jsonObj = response.json()
        jsonObj['quarks'][1]['charge']

        response = requests.post("http://182.18.157.124/ImgVerification/ProjectService.asmx", json={"status":"top","msg":"Model Created Sucessful"})
        response.json()
        
        http://182.18.157.124/ImgVerification/ProjectService.asmx
        
        """

        retJson = {
            "status": 200,
            "msg": "MODEL successfully CREATED"
        }
        return jsonify(retJson)





class Predict(Resource):
    def post(self):
        postedData = request.get_json()
        image_path = postedData["image_path"]
        K.clear_session()

        postedData3 = request.get_json('http://182.18.157.124/ImgVerification/ProjectService.asmx')
        print("posted  data",postedData3)
        postedData2 = requests.get('http://182.18.157.124/ImgVerification/ProjectService.asmx')
        print(postedData2.json)


        # load the class_indices saved in the earlier step
        class_dictionary = np.load('./data1/model/class_indices.npy',allow_pickle=True).item()

        num_classes = len(class_dictionary)

        # add the path to your test image below
        #image_path = './test_images/bin1.jpg'

        orig = cv2.imread(image_path)

        print("[INFO] loading and preprocessing image...")
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        # important! otherwise the predictions will be '0'
        image = image / 255

        image = np.expand_dims(image, axis=0)

        # build the VGG16 network
        model = applications.VGG16(include_top=False, weights='imagenet')

        # get the bottleneck prediction from the pre-trained VGG16 model
        bottleneck_prediction = model.predict(image)

        # build top model
        model = Sequential()
        model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.load_weights(top_model_weights_path)

        # use the bottleneck prediction on the top model to get the final
        # classification
        class_predicted = model.predict_classes(bottleneck_prediction)

        probabilities = model.predict_proba(bottleneck_prediction)

        inID = class_predicted[0]

        inv_map = {v: k for k, v in class_dictionary.items()}

        label = inv_map[inID]

        # get the prediction label
        print("Image ID: {}, Label: {}".format(inID, label))
        response = requests.post("http://182.18.157.124/ImgVerification/ProjectService.asmx",
                                 json={"status": "200", "image id": inID,"msg":label})
        response.json()
        """retJson = {
            "image id": inID,
            "msg": label
        }
        return jsonify(retJson)
        """

api.add_resource(Classify, '/classify')
api.add_resource(Predict, '/predict')

if __name__=="__main__":
    app.run(debug=True,port=8005)
