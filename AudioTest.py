import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import json
import librosa

import argparse
import tensorflow as tf
from models import ACRNN
import pickle
import io
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Speech Emotion Recognition Based on 3D CRNN')
parser.add_argument('--num_classes', default=6, type=int, help='The number of emotion classes.')
parser.add_argument('--batch_size', default=32, type=int, help='The number of samples in each batch.')
parser.add_argument('--audio_dir', default='/content/drive/My Drive/research/youtube_data/audio/', type=str)
parser.add_argument('--label_dir', default='/content/drive/My Drive/research/youtube_data/Name.txt', type=str)
parser.add_argument('--checkpoint', default='/content/drive/My Drive/research/save/', type=str, help='the checkpoint dir')
arg = parser.parse_args()
eps = 1e-5

def read_audio(filepath):
    data, sr = librosa.load(filepath, sr=None)
    return data, sr


def generate_label(emotion):
    # happiness, anger, sadness, frustration and neutral state
    Emo = {'ang': 0, "sad": 1, "hap": 2, "neu": 3, "exc": 4, "fru":5}
    return Emo[emotion]

# def get_emotion(label):
#     emo = {0:'ang', 1:'sad', 2:'hap', 3:'neu', 4:'exc', 5:'fru'}
#     return emo[label]

@tf.function
def predict_step(model, data):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(data, training=False)
    return predictions

def load_label():
    txt_path = arg.label_dir
    data = []
    file_name = []
    file_clips = []
    for line in open(txt_path,"r"):
        if line[-1:] == '\n':
            line = line[:-1]
        tem = line.split(' ')
        file_name.append(tem.pop(0))
        t = [] 
        while tem:
            t.append(float(tem.pop(0)))
            t.append(tem[-1] + float(tem.pop(0)))
        file_clips.append(t)
    return file_name, file_clips

def img_plot(name, timeline, predict):
    file_name, file_clips = load_label()
    for i in range(len(file_name)):
        item_index = name[file_name[i]]
        item_timeline = timeline[item_index]
        item_predict = predict[item_index]
        
        new_timeline = [0]
        new_label = [0]
        cnt = 0
        for j in file_clips[i]:
            if cnt % 2 == 0:
                new_timeline += [j-0.01, j]
                new_label += [0, 1]
            else:
                new_timeline += [j, j + 0.01]
                new_label += [1, 0]
            cnt += 1

        plt.figure()
        plt.title(file_name[i])
        plt.plot(item_timeline, item_predict)
        plt.plot(new_timeline, new_label)
        plt.ylabel('happay probability')
        plt.xlabel('time: sec')
        plt.show()
        

def audio_predict():
    filter_num = 40
    frame = 200
    rootdir = arg.audio_dir
    model = ACRNN(num_classes=arg.num_classes)
    model.load_weights(arg.checkpoint)

    data_name = {}
    data_timeline = []
    data_predict = []
    index = 0
    for item in os.listdir(rootdir):
        data_name[item[:-4]] = index
        index += 1
        item_file = rootdir + item
        data, rate = read_audio(item_file)

        data_num = 8000
        predict_data = np.empty((data_num, 200, filter_num, 3), dtype=np.float32)
        data_num = 0
        mel_spec = ps.logfbank(data, rate, nfilt=filter_num, nfft=int(rate*0.02), winlen=0.02)
        delta1 = ps.delta(mel_spec, 2)
        delta2 = ps.delta(delta1, 2)

        time = mel_spec.shape[0]
        
        if time <= frame:
            part = np.pad(mel_spec, ((0, frame - mel_spec.shape[0]), (0, 0)), 'constant',constant_values=0)
            delta11 = np.pad(delta1,((0,frame-delta1.shape[0]),(0,0)),'constant',constant_values=0)
            delta21 = np.pad(delta2,((0,frame-delta2.shape[0]),(0,0)),'constant',constant_values=0)
            predict_data[data_num, :, :, 0] = part
            predict_data[data_num, :, :, 1] = delta11
            predict_data[data_num, :, :, 2] = delta21
            data_num += 1
        else:
            t = time // frame  
            for i in range(0, t*frame, 100):
                begin = i
                end = begin + frame
                part = mel_spec[begin:end, :]
                delta11 = delta1[begin:end, :]
                delta21 = delta2[begin:end, :]
                predict_data[data_num, :, :, 0] = part
                predict_data[data_num, :, :, 1] = delta11
                predict_data[data_num, :, :, 2] = delta21
                data_num += 1
            if time - end > frame//2:
                part = np.pad(mel_spec[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
                delta11 = np.pad(delta1[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
                delta21 = np.pad(delta2[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
                predict_data[data_num, :, :, 0] = part
                predict_data[data_num, :, :, 1] = delta11
                predict_data[data_num, :, :, 2] = delta21
                data_num += 1
        predict_data = predict_data[:data_num, :, :, :]
    
        # data preprocess, z-score
        mean1 = np.mean(predict_data[:,:,:,0], axis=0)
        mean2 = np.mean(predict_data[:,:,:,1], axis=0)
        mean3 = np.mean(predict_data[:,:,:,2], axis=0)
        std1 = np.std(predict_data[:,:,:,0], axis=0)
        std2 = np.std(predict_data[:,:,:,1], axis=0)
        std3 = np.std(predict_data[:,:,:,2], axis=0)
    
        predict_data[:, :, :, 0] = (predict_data[:, :, :, 0] - mean1)/(std1 + eps)
        predict_data[:, :, :, 1] = (predict_data[:, :, :, 1] - mean2)/(std2 + eps)
        predict_data[:, :, :, 2] = (predict_data[:, :, :, 2] - mean3)/(std3 + eps)

        predict_ds = tf.data.Dataset.from_tensor_slices(predict_data).batch(arg.batch_size)
        predict_label = np.empty((data_num, 6), dtype=np.float32)
        cnt = 0
        for item in predict_ds:
            predict_label[cnt:cnt+item.shape[0]] = predict_step(model, item)
            cnt += item.shape[0]
        hap_pro = predict_label[:, 2]
        timeline = np.arange(0, data_num, 1)
        data_timeline.append(timeline)
        data_predict.append(hap_pro)

        result = np.zeros(data_num)
        for p in range(data_num, 2):
            result[t] = (hap_pro[p] + hap_pro[p+1]) / 2
        np.save(arg.facial_path + 'record/' + item[:-4] + '.npy', result)
    
    img_plot(data_name, data_timeline, data_predict)
    return


if __name__ == '__main__':
    audio_predict()

