import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import json
import librosa

eps = 1e-5

def delete_silent(audioData):
    index = librosa.effects.split(audioData, top_db=30, frame_length=2048, hop_length=512)
    new_x = audioData[index[0][0]:index[0][1]]
    for i in range(1, len(index)):
        new_x = np.append(new_x, audioData[index[i][0]:index[i][1]])
    return new_x

def read_audio(filepath, mode="silent"):
    data, sr = librosa.load(filepath, sr=None)
    # delete little strange at the beginning and the end
    data = data[300:-300]
    if mode == "non-silent":
        data = delete_silent(data)
    return data, sr


def generate_label(emotion):
    # happiness, anger, sadness, frustration and neutral state
    Emo = {'ang': 0, "sad": 1, "hap": 2, "neu": 3, "exc": 4, "fru":5}
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


def read_IEMOCAP():
    data_num = 9109
    filter_num = 40
    frame = 200
    rootdir = 'C:/Users/yzy97/Documents/master/research/data/IEMOCAP_audio'
    pernum = 735
    
    all_label = np.empty((data_num, 1), dtype=np.int8)
    all_data = np.empty((data_num, 200, filter_num, 3), dtype=np.float32)

    data_num = 0
    data_emt = {'ang': 0, "sad": 0, "hap": 0, "neu": 0, "exc": 0, "fru":0}

    f = open('emot_json.json', 'r')
    emot_map = json.load(f)
    for speaker in os.listdir(rootdir):
        sub_dir = os.path.join(rootdir, speaker, 'sentences/wav').replace('\\', '/')
        for sess in os.listdir(sub_dir):
            if sess[7] != 'i':
                continue

            file_dir = os.path.join(sub_dir, sess, '*.wav').replace('\\', '/')
            files = glob.glob(file_dir)
            for filename in files:
                filename = filename.replace('\\', '/')
                wavname = filename.split("/")[-1][:-4]
                emotion = emot_map.get(wavname)
                if emotion:
                    data, rate = read_audio(filename)
                    mel_spec = ps.logfbank(data, rate, nfilt=filter_num, nfft=int(rate*0.02), winlen=0.02)
                    delta1 = ps.delta(mel_spec, 2)
                    delta2 = ps.delta(delta1, 2)
                    # apply zscore
                    em = generate_label(emotion)
                    time = mel_spec.shape[0]
                    
                    if time <= frame:
                        if time > frame // 2:
                            part = np.pad(mel_spec, ((0, frame - mel_spec.shape[0]), (0, 0)), 'constant',constant_values=0)
                            delta11 = np.pad(delta1,((0,frame-delta1.shape[0]),(0,0)),'constant',constant_values=0)
                            delta21 = np.pad(delta2,((0,frame-delta2.shape[0]),(0,0)),'constant',constant_values=0)
                            all_data[data_num, :, :, 0] = part
                            all_data[data_num, :, :, 1] = delta11
                            all_data[data_num, :, :, 2] = delta21

                            all_label[data_num] = em
                            data_emt[emotion] += 1
                            data_num += 1
                    else:
                        t = time // frame
                        if emotion == "hap" or emotion == "ang":
                            for i in range(0, time - frame, 100):
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
                            if time - end > frame // 2:
                                part = np.pad(mel_spec[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
                                delta11 = np.pad(delta1[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
                                delta21 = np.pad(delta2[end:time, :], ((0, frame + end - time), (0, 0)), 'constant', constant_values=0)
                                all_data[data_num, :, :, 0] = part
                                all_data[data_num, :, :, 1] = delta11
                                all_data[data_num, :, :, 2] = delta21

                                all_label[data_num] = em
                                data_emt[emotion] += 1
                                data_num += 1
                        else:    
                            for i in range(0, t*frame, frame):
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


    hap_index = np.arange(data_emt['hap'])
    neu_index = np.arange(data_emt['neu'])
    sad_index = np.arange(data_emt['sad'])
    ang_index = np.arange(data_emt['ang'])
    exc_index = np.arange(data_emt['exc'])
    fru_index = np.arange(data_emt['fru'])
    print(data_emt)
    print(data_num)
    
    hap_cnt, ang_cnt, neu_cnt, sad_cnt, exc_cnt, fru_cnt = 0, 0, 0, 0, 0, 0
    
    for l in range(data_num):
        if all_label[l] == 0:
            ang_index[ang_cnt] = l
            ang_cnt = ang_cnt + 1
        elif all_label[l] == 1:
            sad_index[sad_cnt] = l
            sad_cnt = sad_cnt + 1
        elif all_label[l] == 2:
            hap_index[hap_cnt] = l
            hap_cnt = hap_cnt + 1
        elif all_label[l] == 3:
            neu_index[neu_cnt] = l
            neu_cnt = neu_cnt + 1
        elif all_label[l] == 4:
            exc_index[exc_cnt] = l
            exc_cnt = exc_cnt + 1
        else:
            fru_index[fru_cnt] = l
            fru_cnt = fru_cnt + 1
    
    hap = data_split(hap_index[:pernum])
    ang = data_split(ang_index[:pernum])
    neu = data_split(neu_index[:pernum])
    sad = data_split(sad_index[:pernum])
    exc = data_split(exc_index[:pernum])
    fru = data_split(fru_index[:pernum])
    
    Train_index = np.concatenate((hap[0], ang[0], neu[0], sad[0], exc[0], fru[0]))
    Train_data = all_data[Train_index].copy()
    Train_label = all_label[Train_index].copy()
    Val_index = np.concatenate((hap[1], ang[1], neu[1], sad[1], exc[1], fru[1]))
    Val_data = all_data[Val_index].copy()
    Val_label = all_label[Val_index].copy()
    Test_index = np.concatenate((hap[2], ang[2], neu[2], sad[2], exc[2], fru[2]))
    Test_data = all_data[Test_index].copy()
    Test_label = all_label[Test_index].copy()
    print(Train_data.shape)
    print(Val_data.shape)
    print(Test_data.shape)
    

    output = './IEMOCAP.pkl'
    f = open(output, 'wb')
    pickle.dump((Train_data, Train_label, Val_data, Val_label, Test_data, Test_label), f)
    f.close()
    return


if __name__ == '__main__':
    read_IEMOCAP()
