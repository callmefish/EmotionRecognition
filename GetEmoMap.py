import os
import json
import numpy as np
import pickle
import shutil

class GetEmoMap:
    # create a whole file for IEMOCAP audio data with label
    # the sentence format is extracted from dialog file
    def IEMOCAP(self, rootdir='C:/Users/yzy97/Documents/master/research/data/IEMOCAP_audio'):
        emot_map = {}
        emot_num = {}
        for speaker in os.listdir(rootdir):
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav').replace('\\', '/')
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation').replace('\\', '/')
            for sess in os.listdir(sub_dir):
                emotdir = emoevl + '/' + sess + '.txt'
                with open(emotdir, 'r') as emot_to_read:
                    while True:
                        line = emot_to_read.readline()
                        if not line:
                            break
                        if (line[0] == '['):
                            t = line.split()
                            if t[4] == 'hap' or t[4] == 'neu' or t[4] == 'ang' or t[4] == 'exc' or t[4] == 'sad':
                                if not os.path.exists(sub_dir + '/' + sess + '/' + t[3] + '.wav'):
                                    print(t[3] + ".wav does not exist")
                                    continue
                                if t[4] == 'exc':
                                    emot_map[t[3]] = 'hap'
                                    emot_num['hap'] = emot_num.get('hap', 0) + 1
                                else:
                                    emot_map[t[3]] = t[4]
                                    emot_num[t[4]] = emot_num.get(t[4], 0) + 1
        print(emot_num)
        return emot_map

if __name__=="__main__":
    a = GetEmoMap()
    res = a.IEMOCAP()
    emot_json = json.dumps(res)
    if os.path.exists('emot_json_IEMOCAP_4.json'):
        os.remove('emot_json_IEMOCAP_4.json')
    f = open('emot_json_IEMOCAP_4.json', 'w')
    f.write(emot_json)