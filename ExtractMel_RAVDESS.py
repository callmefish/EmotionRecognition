import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import json
import librosa
import matplotlib.pyplot as plt

eps = 1e-5

def read_audio(filepath):
    data, sr = librosa.load(filepath, sr=None)
    # Trim leading and trailing silence from an audio signal.
    data_trimmed, index = librosa.effects.trim(data, top_db=25, frame_length=4096, hop_length=2048)
    
    # plot the area of trim
    # plt.figure()
    # plt.plot(data)
    # plt.axvspan(index[0], index[1], alpha=0.5, color='red')
    # plt.show()
    # plt.close()
    return data_trimmed, sr


def generate_label(emotion):
    '''
        {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
    '''
    Emo = {"01": 0, "02": 1, "03": 2, "04": 3, "05": 4, "06":5, "07":6,"08":7}
    return Emo[emotion]

def data_split(array):
    np.random.shuffle(array)
    L = len(array)
    train_len = int(0.6 * L)
    train_index = array[:train_len]
    val_len = int(0.2 * L)
    val_index = array[train_len: train_len + val_len]
    test_index = array[train_len + val_len:]
    return [train_index, val_index, test_index]


def read_RAVDESS():
    data_num = 10000
    frame = 100
    filter_num = 40
    rootdir = 'C:/Users/yzy97/Documents/master/research/data/RAVDESS_audio/'
    pernum = 560
    
    all_label = np.empty((data_num, 1), dtype=np.int8)
    all_data = np.empty((data_num, frame, filter_num, 3), dtype=np.float32)

    data_num = 0
    data_emt = {"01": 0, "02": 0, "03": 0, "04": 0, "05": 0, "06":0, "07":0,"08":0}

    actors = os.listdir(rootdir)
    cnt1 = 0
    cnt2 = 0
    for actor in actors:
        sub_dir = rootdir + actor
        audios = os.listdir(sub_dir)
        for audio in audios:
            temp = audio.split('-')
            emotion = temp[2]
            vocal = temp[1]
            # without song
            if vocal == '01':
                audio_path = sub_dir + '/' + audio
                data, rate = read_audio(audio_path)
                mel_spec = ps.logfbank(data, rate, nfilt=filter_num, nfft=int(rate*0.02), winlen=0.02)
                delta1 = ps.delta(mel_spec, 2)
                delta2 = ps.delta(delta1, 2)

                em = generate_label(emotion)
                time = mel_spec.shape[0]
        
                if time <= frame:
                    part = np.pad(mel_spec, ((0, frame - mel_spec.shape[0]), (0, 0)), 'constant',constant_values=0)
                    delta11 = np.pad(delta1,((0,frame-delta1.shape[0]),(0,0)),'constant',constant_values=0)
                    delta21 = np.pad(delta2,((0,frame-delta2.shape[0]),(0,0)),'constant',constant_values=0)
                    all_data[data_num, :, :, 0] = part
                    all_data[data_num, :, :, 1] = delta11
                    all_data[data_num, :, :, 2] = delta21
                    all_label[data_num] = em
                    data_emt[emotion] += 1
                    data_num += 1
                    cnt1 += 1
                else:
                    t = time // frame  
                    for i in range(0, time - frame + 1, frame//2):
                        begin = i
                        end = begin + frame
                        part = mel_spec[begin:end, :]
                        delta11 = delta1[begin:end, :]
                        delta21 = delta2[begin:end, :]
                        all_data[data_num, :, :, 0] = part
                        all_data[data_num, :, :, 1] = delta11
                        all_data[data_num, :, :, 2] = delta21
                        all_label[data_num] = em
                        data_emt[emotion] += 1
                        data_num += 1
                    if time - end > frame//2:
                        part = np.pad(mel_spec[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
                        delta11 = np.pad(delta1[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
                        delta21 = np.pad(delta2[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
                        all_data[data_num, :, :, 0] = part
                        all_data[data_num, :, :, 1] = delta11
                        all_data[data_num, :, :, 2] = delta21
                        all_label[data_num] = em
                        data_emt[emotion] += 1
                        data_num += 1
                        cnt2 += 1
                    else:
                        part = mel_spec[time-frame:time, :]
                        delta11 = delta1[time-frame:time, :]
                        delta21 = delta2[time-frame:time, :]
                        all_data[data_num, :, :, 0] = part
                        all_data[data_num, :, :, 1] = delta11
                        all_data[data_num, :, :, 2] = delta21
                        all_label[data_num] = em
                        data_emt[emotion] += 1
                        data_num += 1
    all_data = all_data[:data_num, :, :, :]
    all_label = all_label[:data_num]      

    # data preprocess, z-score
    mean1 = np.mean(all_data[:,:,:,0], axis=0)
    mean2 = np.mean(all_data[:,:,:,1], axis=0)
    mean3 = np.mean(all_data[:,:,:,2], axis=0)
    std1 = np.std(all_data[:,:,:,0], axis=0)
    std2 = np.std(all_data[:,:,:,1], axis=0)
    std3 = np.std(all_data[:,:,:,2], axis=0)
    all_data[:, :, :, 0] = (all_data[:, :, :, 0] - mean1)/(std1 + eps)
    all_data[:, :, :, 1] = (all_data[:, :, :, 1] - mean2)/(std2 + eps)
    all_data[:, :, :, 2] = (all_data[:, :, :, 2] - mean3)/(std3 + eps)

    neu_index = np.arange(data_emt['01'])
    cal_index = np.arange(data_emt['02'])
    hap_index = np.arange(data_emt['03'])
    sad_index = np.arange(data_emt['04'])
    ang_index = np.arange(data_emt['05'])
    fea_index = np.arange(data_emt['06'])
    dis_index = np.arange(data_emt['07'])
    sur_index = np.arange(data_emt['08'])
    print(data_emt)
    
    neu_cnt, cal_cnt, hap_cnt, sad_cnt, ang_cnt, fea_cnt, dis_cnt, sur_cnt = 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(data_num):
        if all_label[i] == 0:
            neu_index[neu_cnt] = i
            neu_cnt += 1
        elif all_label[i] == 1:
            cal_index[cal_cnt] = i
            cal_cnt += 1
        elif all_label[i] == 2:
            hap_index[hap_cnt] = i
            hap_cnt += 1
        elif all_label[i] == 3:
            sad_index[sad_cnt] = i
            sad_cnt += 1
        elif all_label[i] == 4:
            ang_index[ang_cnt] = i
            ang_cnt += 1
        elif all_label[i] == 5:
            fea_index[fea_cnt] = i
            fea_cnt += 1
        elif all_label[i] == 6:
            dis_index[dis_cnt] == i
            dis_cnt += 1
        else:
            sur_index[sur_cnt] = i
            sur_cnt += 1
    
    neu = data_split(neu_index)
    cal = data_split(cal_index)
    hap = data_split(hap_index)
    sad = data_split(sad_index)
    ang = data_split(ang_index)
    fea = data_split(fea_index)
    dis = data_split(dis_index)
    sur = data_split(sur_index)
    
    Train_index = np.concatenate((neu[0], cal[0], hap[0], sad[0], ang[0], fea[0], dis[0], sur[0]))
    Train_data = all_data[Train_index].copy()
    Train_label = all_label[Train_index].copy()
    Val_index = np.concatenate((neu[1], cal[1], hap[1], sad[1], ang[1], fea[1], dis[1], sur[1]))
    Val_data = all_data[Val_index].copy()
    Val_label = all_label[Val_index].copy()
    Test_index = np.concatenate((neu[2], cal[2], hap[2], sad[2], ang[2], fea[2], dis[2], sur[2]))
    Test_data = all_data[Test_index].copy()
    Test_label = all_label[Test_index].copy()
    print(Train_data.shape)
    print(Val_data.shape)
    print(Test_data.shape)
    

    output = './RAVDESS.pkl'
    f = open(output, 'wb')
    pickle.dump((Train_data, Train_label, Val_data, Val_label, Test_data, Test_label), f)
    f.close()
    return


if __name__ == '__main__':
    read_RAVDESS()

