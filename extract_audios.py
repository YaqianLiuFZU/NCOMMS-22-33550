import os
import librosa
import numpy as np
import soundfile

time = 4
dir = 'D:/dataset/audio'
list = os.listdir(dir)
for j in range(len(list)):
    audios = librosa.core.load(os.path.join(list[j]), sr=22050)
	
    y = audios[0]
    sr = audios[1]
    length = int(sr * time)
    if len(y) < length:
        y = np.array(list(y) + [0 for i in range(length - len(y))])
    else:
        remain = len(y) - length
        y = y[remain//2:-(remain - remain//2)]
	    
    soundfile.write(os.path.join(list[j]+'_croppad.wav'), y, sr)

