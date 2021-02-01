import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import python_speech_features as ps
import json

eps = 1e-5

def draw_silent(raw_data, silent_index):
    fig, ax = plt.subplots()
    ax.plot(raw_data)
    for i in silent_index:
        ax.axvspan(i[0], i[1], alpha=0.5, color='red')
    plt.show()

    plt.close()
    return

def draw_data(raw_data):
    plt.plot(raw_data)
    plt.show()
    plt.close()
    return

f = open('emot_json_IEMOCAP_4.json', 'r')
emot_map = json.load(f)
keys = list(emot_map.keys())[:241]

base_filepath = "C:/Users/yzy97/Documents/master/research/data/IEMOCAP_audio/Session1/sentences/wav/"
frame_length = 2048
hop_length = 512
top_list = [20, 15, 10, 9, 8, 7, 6, 5]

silent, sr = librosa.load(base_filepath + "Ses01F_impro01/Ses01F_impro01_F001.wav", sr=None)
silent = silent[:10000]

for key in keys:
    # print(key)
    sub = key.split("_")
    filepath = base_filepath + sub[0] + "_" + sub[1] + "/" + key + ".wav"
    data, sr = librosa.load(filepath, sr=None)
    max_value = np.max(data)
    for top_db in top_list:
        index1 = librosa.effects.split(data, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
        if len(index1) == 1 and (index1[0][1] - index1[0][0])/len(data)>=0.8:
            continue
        if len(data) < sr:
            index2 = [[0, len(data)]]
            new_data = data
            break
        index2 = [index1[0]]
        for i in range(1, len(index1)):
            if index1[i][0] - index2[-1][1] <= frame_length:
                index2[-1][1] = index1[i][1]
            else:
                index2.append(index1[i])
        length = 0
        length_list = []
        index3 = []
        for j in range(len(index2)):
            max_temp = np.max(data[index2[j][0]:index2[j][1]])
            if max_temp < max_value/3:
                continue
            index3.append(index2[j])
            length += index2[j][1]-index2[j][0]
            length_list.append(index2[j][1]-index2[j][0])
        index2 = index3
        
        if length < sr:
            max_index = length_list.index(max(length_list))
            part = (sr - length + 200) // 2
            index2[max_index] = [max(0, index2[max_index][0]-part), min(index2[max_index][1]+part, len(data))]
        else:
            if len(index2) == 1 and (index2[0][1] - index2[0][0])/len(data)>=0.7:
                continue
        new_data = data[index2[0][0]:index2[0][1]]
        for j in range(1, len(index2)):
            new_data = np.append(new_data, data[index2[j][0]:index2[j][1]])
        break
    if len(new_data) < sr:
        item_len = (sr - len(new_data)+200)//2
        item = silent[:item_len]
        new_data = np.concatenate((item, new_data, item))
    # draw_silent(data, index2)
    
            
