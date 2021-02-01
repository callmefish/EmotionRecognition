from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

parser = argparse.ArgumentParser()
parser.add_argument('--dir_prefix', default='/content/drive/MyDrive/research/', type=str)
parser.add_argument('--facial_path', default='/content/drive/MyDrive/research/facial/', type=str)
arg = parser.parse_args()


def make_sure_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    # parameters for loading data and images
    detection_model_path = arg.facial_path + 'haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = arg.facial_path + 'models/_mini_XCEPTION.102-0.66.hdf5'

    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    # getting input model shapes for inference
    # emotion_target_size = emotion_classifier.input_shape
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    video_path = arg.dir_prefix + "youtube_data/video/"
    video_list = os.listdir(video_path)
    

    for video_name in video_list:
        print("The video is " + video_name)
        make_sure_path('record/')
        video_item_path = video_path + video_name
        cap = cv2.VideoCapture(video_item_path)

        start_time = time.time()
        if cap.isOpened():
            rate = int(cap.get(5)) + 1
            FrameNumber = cap.get(7)
            duration = FrameNumber / rate
            # width = cap.get(3)
            # height = cap.get(4)
            print("The duration is %f s" % duration)
            print("The number of frame is %d " % FrameNumber)
            print("The rate of video is %d " % rate)

            happy = []
            img = []
            cnt = 0
            index = 0
            non_face = []
            while True:
                res, frame = cap.read()
                if index >= 900:
                    print("None of images")
                    img = np.array(img) / 255.0
                    img = np.expand_dims(img, -1)
                    preds = emotion_classifier.predict(img)
                    happy.append(preds[:, 3])
                    break
                if not res:
                    print("None of images")
                    img = np.array(img) / 255.0
                    img = np.expand_dims(img, -1)
                    preds = emotion_classifier.predict(img)
                    happy.append(preds[:, 3])
                    break
                else:
                    frame = cv2.resize(frame, (300, 250))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                            flags=cv2.CASCADE_SCALE_IMAGE)
                    if len(faces) > 0:
                        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                        (fX, fY, fW, fH) = faces
                        roi = gray[fY:fY + fH, fX:fX + fW]
                        roi = cv2.resize(roi, (64, 64))
                        if cnt < 64:
                            img.append(roi)
                            cnt += 1
                        else:
                            img = np.array(img) / 255.0
                            img = np.expand_dims(img, -1)
                            preds = emotion_classifier.predict(img)
                            happy.append(preds[:, 3])
                            img = [roi]
                            cnt = 1
                    else:
                        non_face.append(index)
                index += 1
            happy = np.concatenate(happy)
            for i in non_face:
                happy = np.insert(happy, i, 0)
            t = 0
            result = np.zeros(index // rate + 1)
            for p in range(0, index, rate):
                result[t] = happy[p:p+rate].mean()
                t += 1
            np.save(arg.facial_path + 'record/' + video_name[:-4] + '.npy', result)
            
            result_len = result.shape[0]
            result = result.repeat(2)

            x_axis = np.arange(0, result_len, dtype=float)
            x_axis = x_axis.reshape(result_len, 1)
            x_axis = x_axis.repeat(2, axis=1)
            x_axis[:, 1] += 0.99
            x_axis = x_axis.reshape(2 * result_len)

            plt.figure()
            plt.plot(x_axis, result)
            plt.legend(['predict', 'actual'])
            plt.ylabel('Probability of happy')
            plt.xlabel('time / s')
            plt.savefig(arg.facial_path + 'record/' + video_name[:-4] + '.png')

        cap.release()

        end_time = time.time()
        print("Spending time is %s s" % (end_time - start_time))
