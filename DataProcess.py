import librosa
import numpy as np
import json
import os
import glob
import python_speech_features as ps
import pickle
import matplotlib.pyplot as plt

eps = 1e-5

class data_process:
    def __init__(self):
        self.frame_length = 2048
        self.hop_length = 512
        self.top_list = [20, 15, 10, 9, 8, 7, 6, 5]
        self.pernum = 100
        self.filter_num = 40
        self.rootdir = 'C:/Users/yzy97/Documents/master/research/data/IEMOCAP_audio'
        silent, sr = librosa.load(self.rootdir + "/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F001.wav", sr=None)
        self.silent = silent[:10000]

    # draw the audio data with silent part
    def draw_silent(self, raw_data, silent_index):
        fig, ax = plt.subplots()
        ax.plot(raw_data)
        for i in silent_index:
            ax.axvspan(i[0], i[1], alpha=0.5, color='red')
        plt.show()
        return 

    def draw_mel(self, matrix):
        plt.imshow(matrix)
        plt.show()
        plt.close()
        return 

    def delete_silent(self, audioData, sr):
        max_value = np.max(audioData)
        index2 = []

        # dynamic select top_db
        for top_db in self.top_list:
            # if length of audioData < sr, break the for loop
            if len(audioData) < sr:
                index2 = [[0, len(audioData)]]
                new_data = audioData
                break
            
            # initial index of non-silent parts
            index1 = librosa.effects.split(audioData, top_db=top_db, frame_length=self.frame_length, hop_length=self.hop_length)
            
            # if non-silent == all data, continue and choose smaller top_db
            if len(index1) == 1 and (index1[0][1] - index1[0][0])/len(audioData)>=0.8:
                continue
            
            # create index2, which merge some parts of non-silent
            index2 = [index1[0]]
            for i in range(1, len(index1)):
                if index1[i][0] - index2[-1][1] <= self.frame_length:
                    index2[-1][1] = index1[i][1]
                else:
                    index2.append(index1[i])
            # compute each length and the whole length of non-silent part
            length = 0
            length_list = []
            index3 = []
            for j in range(len(index2)):
                # delete the part which max value is too small, seem it as silent part
                max_temp = np.max(audioData[index2[j][0]:index2[j][1]])
                if max_temp < max_value/3:
                    continue
                
                index3.append(index2[j])
                length += index2[j][1]-index2[j][0]
                length_list.append(index2[j][1]-index2[j][0])
            # new index2 with suitable non-silent part
            index2 = index3
            
            # if whole length of non-silent part < sr, 
            # expand index for the part with the largest range
            if length < sr:
                max_index = length_list.index(max(length_list))
                part = (sr - length + 200) // 2
                index2[max_index] = [max(0, index2[max_index][0]-part), min(index2[max_index][1]+part, len(audioData))]
            else:
                if len(index2) == 1 and (index2[0][1] - index2[0][0])/len(audioData)>=0.7:
                    continue
            break
        
        if not index2:
            index2 = index1
        # merge all of non-silent part
        new_data = audioData[index2[0][0]:index2[0][1]]
        for j in range(1, len(index2)):
            new_data = np.append(new_data, audioData[index2[j][0]:index2[j][1]])
        # if length of new_data < sr, add the example silent part to expand data
        if len(new_data) < sr + 200:
            item_len = (sr - len(new_data) + 200) // 2
            item = self.silent[:item_len]
            new_data = np.concatenate((item, new_data, item))    
        return new_data

    def read_audio(self, filepath, mode="non-silent"):
        data, sr = librosa.load(filepath, sr=None)
        # delete silent part
        if mode == "non-silent":
            data = self.delete_silent(data, sr)
        time = np.arange(0, len(data)) * (1.0 / sr)
        return data, time, sr

    def generate_label(self, emotion):
        # happiness, anger, sadness and neutral state
        Emo = {"hap": 0, "ang": 1, "sad": 2, "neu": 3}
        return Emo[emotion]
    

    def putDataInTrainSet(self, Traindata1, Traindata2, Traindata3, Part, Delta11, Delta21, Train_num):
        Traindata1[Train_num * self.pernum:(Train_num + 1) * self.pernum] = Part
        Traindata2[Train_num * self.pernum:(Train_num + 1) * self.pernum] = Delta11
        Traindata3[Train_num * self.pernum:(Train_num + 1) * self.pernum] = Delta21
        return Traindata1, Traindata2, Traindata3


    def data_split(self, array):
        np.random.shuffle(array)
        L = len(array)
        train_len = int(0.6 * L)
        train_index = array[:train_len]
        val_len = int(0.2 * L)
        val_index = array[train_len: train_len + val_len]
        test_index = array[train_len + val_len:]
        return [train_index, val_index, test_index]

    def read_IEMOCAP(self):
        # initialize the numpy array for storing mel, delta1, delta2
        data_num = 20000
        all_label = np.empty((data_num, 1), dtype=np.int8)
        all_data = np.empty((data_num, self.pernum, self.filter_num, 3), dtype=np.float32)
        data_num = 0
        data_emt = {'ang': 0, "sad": 0, "hap": 0, "neu": 0}

        # read json file
        f = open('emot_json_IEMOCAP_4.json', 'r')
        emot_map = json.load(f)
        f.close()

        rootdir_sub = os.listdir(self.rootdir)
        for speaker in rootdir_sub:
            print('-----------------------')
            print(speaker)
            sub_dir = os.path.join(self.rootdir, speaker, 'sentences/wav').replace('\\', '/')
            for sess in os.listdir(sub_dir):
                # # ignore script part of data
                # if (sess[7] != 'i'):
                #     continue
                file_dir = os.path.join(sub_dir, sess, '*.wav').replace('\\', '/')
                files = glob.glob(file_dir)
                for filename in files:
                    filename = filename.replace('\\', '/')
                    # print(filename)
                    wavname = filename.split("/")[-1][:-4]
                    emotion = emot_map.get(wavname)
                    if emotion:
                        data, time, rate = self.read_audio(filename)
                        # Compute log Mel-filterbank energy features from an audio signal.
                        mel_spec = ps.logfbank(data, rate, nfilt=self.filter_num)
                        # Compute delta features from a feature vector sequence.
                        delta1 = ps.delta(mel_spec, 2)
                        delta2 = ps.delta(delta1, 2)
                        # self.draw_mel(delta1.T)

                        time = mel_spec.shape[0]
                        em = self.generate_label(emotion)
                       
                        t = time // self.pernum
                        for i in range(0, t*self.pernum, self.pernum):
                            begin = i
                            end = begin + self.pernum
                            part = mel_spec[begin:end, :]
                            if len(part) < self.pernum:
                                print(time)
                                print('----')
                            delta11 = delta1[begin:end, :]
                            delta21 = delta2[begin:end, :]
                            all_data[data_num, :, :, 0] = part
                            all_data[data_num, :, :, 1] = delta11
                            all_data[data_num, :, :, 2] = delta21
                            all_label[data_num] = em
                            
                            data_num += 1
                            data_emt[emotion] += 1
                        if time - end > self.pernum // 3:
                            begin = time - self.pernum
                            end = time
                            part = mel_spec[begin:end, :]
                            delta11 = delta1[begin:end, :]
                            delta21 = delta2[begin:end, :]
                            
                            all_data[data_num, :, :, 0] = part
                            all_data[data_num, :, :, 1] = delta11
                            all_data[data_num, :, :, 2] = delta21
                            all_label[data_num] = em
                            
                            data_num += 1
                            data_emt[emotion] += 1

        all_data = all_data[:data_num, :, :, :]
        all_label = all_label[:data_num]  
        print(data_num)

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

        # shuffle the data, and split it into train, val, test
        hap_index = np.arange(data_emt['hap'])
        ang_index = np.arange(data_emt['ang'])
        sad_index = np.arange(data_emt['sad'])
        neu_index = np.arange(data_emt['neu'])
        print(data_emt)

        hap_cnt, ang_cnt, sad_cnt, neu_cnt = 0, 0, 0, 0

        for i in range(data_num):
            if all_label[i] == 0:
                hap_index[hap_cnt] = i
                hap_cnt += 1
            elif all_label[i] == 1:
                ang_index[ang_cnt] = i
                ang_cnt += 1
            elif all_label[i] == 2:
                sad_index[sad_cnt] = i
                sad_cnt += 1
            elif all_label[i] == 3:
                neu_index[neu_cnt] = i
                neu_cnt += 1

        hap = self.data_split(hap_index)
        ang = self.data_split(ang_index)
        sad = self.data_split(sad_index)
        neu = self.data_split(neu_index)
        
        Train_index = np.concatenate((hap[0], ang[0], sad[0], neu[0]))
        Train_data = all_data[Train_index].copy()
        Train_label = all_label[Train_index].copy()
        Val_index = np.concatenate((hap[1], ang[1], sad[1], neu[1]))
        Val_data = all_data[Val_index].copy()
        Val_label = all_label[Val_index].copy()
        Test_index = np.concatenate((hap[2], ang[2], sad[2], neu[2]))
        Test_data = all_data[Test_index].copy()
        Test_label = all_label[Test_index].copy()

        print(Train_data.shape)
        print(Val_data.shape)
        print(Test_data.shape)

        # output = './IEMOCAP_4.pkl'
        # f = open(output, 'wb')
        # pickle.dump((Train_data, Train_label, Val_data, Val_label, Test_data, Test_label), f)
        # f.close()
        return

if __name__ == "__main__":
    # read_IEMOCAP()
    process = data_process()
    process.read_IEMOCAP()