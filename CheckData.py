import os
import glob
import json

class check_audio:
    def IEMOCAP(self, rootdir):
        rootdir = 'C:/Users/yzy97/Documents/master/research/data/IEMOCAP_audio'
        cnt = 0
        f = open('emot_json.json', 'r')
        emot_map = json.load(f)
        f.close()
        print(set(emot_map.values()))
        emo_cnt = {'exc':0, 'sad':0, 'neu':0, 'hap':0, 'ang':0, 'fru':0}
        for speaker in os.listdir(rootdir):
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav').replace('\\', '/')
            for sess in os.listdir(sub_dir):
                if (sess[7] != 'i'):
                    continue
                file_dir = os.path.join(sub_dir, sess, '*.wav').replace('\\', '/')
                files = glob.glob(file_dir)
                for filename in files:
                    filename = filename.replace('\\', '/')
                    wavname = filename.split("/")[-1][:-4]
                    emotion = emot_map.get(wavname)
                    if emotion:
                        cnt += 1
                        emo_cnt[emotion] += 1


        print(cnt)
        print(emo_cnt)
        cnt = 0
        for i, j in emot_map.items():
            if i[:5] != 'Ses05':
                cnt += 1
        print(cnt)
        # print(list(emot_map.values()).count('hap'))

    def RAVDESS(self):
        rootdir = "C:/Users/yzy97/Documents/master/research/data/RAVDESS_audio/"
        emo_label = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
                    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
        emo_map = {}
        actors = os.listdir(rootdir)
        for actor in actors:
            sub_dir = rootdir + actor
            audios = os.listdir(sub_dir)
            for audio in audios:
                temp = audio.split('-')
                emotion = temp[2]
                vocal = temp[1]
                # without song
                if vocal == '01':
                    emo_map[emo_label[emotion]] = emo_map.get(emo_label[emotion], 0) + 1
        print(emo_map)


if __name__ == "__main__":
    check_RAVDESS()
    