import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import json
import librosa

import argparse
# import tensorflow as tf
# from models import ACRNN
import pickle
import io
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Speech Emotion Recognition Based on 3D CRNN')
parser.add_argument('--num_classes', default=8, type=int, help='The number of emotion classes.')
parser.add_argument('--batch_size', default=128, type=int, help='The number of samples in each batch.')
parser.add_argument('--audio_dir', default='youtube_data/audio/', type=str)
parser.add_argument('--label_dir', default='youtube_data/Name.txt', type=str)
parser.add_argument('--checkpoint', default='save/', type=str, help='the checkpoint dir')
parser.add_argument('--speech_path', default='speech/', type=str)
arg = parser.parse_args()
eps = 1e-5

def read_audio(filepath):
    data, sr = librosa.load(filepath, sr=None)
    index = librosa.effects.split(data, top_db=25, frame_length=4096, hop_length=512)
    print("the length of raw data is {}".format(len(data)))
    print("the duration of raw data is {} sec".format(len(data)/sr))
    # plot the area of split
    name = filepath.split('/')[-1]
    name = name[:-4]
    plt.figure(figsize=(40,5))
    plt.plot(data[:index[14][1]])
    for x, y in index[:15]:
        plt.axvspan(x, y, alpha=0.5, color='red')
    plt.savefig(name + '.png')
    plt.show()
    plt.close()

    return data, index, sr

# @tf.function
# def predict_step(model, data):
#     # training=False is only needed if there are layers with different
#     # behavior during training versus inference (e.g. Dropout).
#     predictions = model(data, training=False)
#     return predictions

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
            t.append(t[-1] + float(tem.pop(0)))
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
        plt.savefig(file_name[i] + '.png')
        plt.show()

def img_plot2(name, timeline, predict):
    for k,v in name.items():
        item_timeline = timeline[v]
        item_predict = predict[v]

        plt.figure()
        plt.title(k)
        plt.plot(item_timeline, item_predict)
        plt.ylabel('happay probability')
        plt.xlabel('time: sec')
        plt.savefig(k + '.png')
        plt.show()
        

def audio_predict():
    filter_num = 40
    frame = 100
    rootdir = arg.audio_dir
    # model = ACRNN(num_classes=arg.num_classes)
    # model.load_weights(arg.checkpoint)

    data_name = {}
    data_timeline = []
    data_predict = []
    index = 0
    for item in os.listdir(rootdir):
        audio_name = item[:-4]
        print(item)
        data_name[item[:-4]] = index
        index += 1
        item_file = rootdir + item
        data, index, rate = read_audio(item_file)
        break

        # data_num = 8000
        # predict_data = np.empty((data_num, frame, filter_num, 3), dtype=np.float32)
        # predict_time = np.empty((data_num, 1), dtype=np.float32)
        # data_num = 0
        # pre_data_num = 0
        # result = np.zeros(1)

        # for x, y in index:
        #     mel_spec = ps.logfbank(data[x:y], rate, nfilt=filter_num, nfft=int(rate*0.02), winlen=0.02)
        #     delta1 = ps.delta(mel_spec, 2)
        #     delta2 = ps.delta(delta1, 2)

        #     time = mel_spec.shape[0]
        #     if time < frame // 2:
        #       continue
        
        #     if time <= frame:
        #         predict_time[data_num] = x * 1.0 / rate
        #         part = np.pad(mel_spec, ((0, frame - mel_spec.shape[0]), (0, 0)), 'constant',constant_values=0)
        #         delta11 = np.pad(delta1,((0,frame-delta1.shape[0]),(0,0)),'constant',constant_values=0)
        #         delta21 = np.pad(delta2,((0,frame-delta2.shape[0]),(0,0)),'constant',constant_values=0)
        #         predict_data[data_num, :, :, 0] = part
        #         predict_data[data_num, :, :, 1] = delta11
        #         predict_data[data_num, :, :, 2] = delta21
        #         data_num += 1
        #     else:

        #         for i in range(0, time-frame+1, frame//2):
        #             predict_time[data_num] = x * 1.0 / rate + 0.5 * (i//(frame//2))
        #             begin = i
        #             end = begin + frame
        #             part = mel_spec[begin:end, :]
        #             delta11 = delta1[begin:end, :]
        #             delta21 = delta2[begin:end, :]
        #             predict_data[data_num, :, :, 0] = part
        #             predict_data[data_num, :, :, 1] = delta11
        #             predict_data[data_num, :, :, 2] = delta21
        #             data_num += 1
        #         if time - end > frame//2:
        #             predict_time[data_num] = predict_time[data_num-1] + 0.5
        #             part = np.pad(mel_spec[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
        #             delta11 = np.pad(delta1[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
        #             delta21 = np.pad(delta2[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
        #             predict_data[data_num, :, :, 0] = part
        #             predict_data[data_num, :, :, 1] = delta11
        #             predict_data[data_num, :, :, 2] = delta21
        #             data_num += 1
        #         # else:
        #         #     part = mel_spec[time-frame:time, :]
        #         #     delta11 = delta1[time-frame:time, :]
        #         #     delta21 = delta2[time-frame:time, :]
        #         #     predict_data[data_num, :, :, 0] = part
        #         #     predict_data[data_num, :, :, 1] = delta11
        #         #     predict_data[data_num, :, :, 2] = delta21
        #         #     data_num += 1
        #     tem_predict_data = predict_data[pre_data_num:data_num, :, :, :]
        
        #     # data preprocess, z-score
        #     mean1 = np.mean(tem_predict_data[:,:,:,0], axis=0)
        #     mean2 = np.mean(tem_predict_data[:,:,:,1], axis=0)
        #     mean3 = np.mean(tem_predict_data[:,:,:,2], axis=0)
        #     std1 = np.std(tem_predict_data[:,:,:,0], axis=0)
        #     std2 = np.std(tem_predict_data[:,:,:,1], axis=0)
        #     std3 = np.std(tem_predict_data[:,:,:,2], axis=0)
        
        #     tem_predict_data[:, :, :, 0] = (tem_predict_data[:, :, :, 0] - mean1)/(std1 + eps)
        #     tem_predict_data[:, :, :, 1] = (tem_predict_data[:, :, :, 1] - mean2)/(std2 + eps)
        #     tem_predict_data[:, :, :, 2] = (tem_predict_data[:, :, :, 2] - mean3)/(std3 + eps)

            # tem_predict_ds = tf.data.Dataset.from_tensor_slices(tem_predict_data).batch(arg.batch_size)
            # tem_predict_label = np.empty((data_num-pre_data_num, arg.num_classes), dtype=np.float32)
            # cnt = 0
            # for item in tem_predict_ds:
            #     tem_predict_label[cnt:cnt+item.shape[0]] = predict_step(model, item)
            #     cnt += item.shape[0]
            # hap_pro = tem_predict_label[:, 2]
            # print(hap_pro.shape)
            # pre_data_num = data_num
            # result = np.concatenate((result, hap_pro))

            # print(data_num)
            # result = np.zeros(data_num//2 + 1)
            # print(result.shape)
            # t = 0
            # for p in range(0,data_num-1, 2):
            #     result[t] = (hap_pro[p] + hap_pro[p+1]) / 2
            #     t += 1
        # result = result[1:]
        # result = np.expand_dims(result, axis=1)
        predict_time = predict_time[:data_num]
        dif = 1000
        print("shape of predict_time = {}".format(predict_time.shape))
        for i in range(1, predict_time.shape[0]):
          if predict_time[i][0] < predict_time[i-1][0]:
            print('overlap')
            print("cur = {}, pre = {}".format(predict_time[i][0],predict_time[i-1][0]))
          dif = min(dif, predict_time[i][0] - predict_time[i-1][0])
        print(dif)
        print(type(dif))
        break
        # result = np.concatenate((result, predict_time), axis=1)
        # print("the shape of result = {}".format(result.shape))
        # np.save(arg.speech_path + 'record/data/' + audio_name + '.npy', result)
        # print('======================')
            # timeline = np.arange(0, data_num, 1)
            # data_timeline.append(timeline)
            # data_predict.append(hap_pro)
    
    # img_plot(data_name, data_timeline, data_predict)
    return


if __name__ == '__main__':
    audio_predict()

