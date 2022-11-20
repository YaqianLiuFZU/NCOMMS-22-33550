import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

device = torch.device('cuda')
mtcnn = MTCNN(image_size=(720, 1280), device=device)

length=3.6
frame=30
fps=60
video_dir='D:/dataset/video/'

failed_videos = []

s_d = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
n_processed = 0
for sess in tqdm(sorted(os.listdir(video_dir))):   
    for filename in os.listdir(os.path.join(video_dir)):
           
        if filename.endswith('.avi'):
                        
            cap = cv2.VideoCapture(os.path.join(video_dir, filename))  
            #calculate length in frames
            framen = 0
            while True:
                i,q = cap.read()
                if not i:
                    break
                framen += 1
            cap = cv2.VideoCapture(os.path.join(video_dir, filename))

            if length*fps > framen:                    
                skip_begin = int((framen - (length*fps)) // 2)
                for i in range(skip_begin):
                    _, im = cap.read() 
                    
            framen = int(length*fps)    
            frames_to_select = s_d(frame,framen)
            save_fps = frame // (framen // fps) 
            out = cv2.VideoWriter(os.path.join(video_dir, filename[:-4]+'_facecroppad.avi'),cv2.VideoWriter_fourcc('M','J','P','G'), save_fps, (224,224))
            numpy_video = []
            success = 0
            frame_ctr = 0
            
            while True: 
                ret, im = cap.read()
                if not ret:
                    break
                if frame_ctr not in frames_to_select:
                    frame_ctr += 1
                    continue
                else:
                    frames_to_select.remove(frame_ctr)
                    frame_ctr += 1

                try:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                except:
                    failed_videos.append((sess, i))
                    break
	    
                temp = im[:,:,-1]
                im_rgb = im.copy()
                im_rgb[:,:,-1] = im_rgb[:,:,0]
                im_rgb[:,:,0] = temp
                im_rgb = torch.tensor(im_rgb)
                im_rgb = im_rgb.to(device)

                bbox = mtcnn.detect(im_rgb)
                if bbox[0] is not None:
                    bbox = bbox[0][0]
                    bbox = [round(x) for x in bbox]
                    x1, y1, x2, y2 = bbox
                im = im[y1:y2, x1:x2, :]
                im = cv2.resize(im, (224,224))
                out.write(im)
                numpy_video.append(im)
            if len(frames_to_select) > 0:
                for i in range(len(frames_to_select)):
                    out.write(np.zeros((224,224,3), dtype = np.uint8))
                    numpy_video.append(np.zeros((224,224,3), dtype=np.uint8))
            out.release() 
            np.save(os.path.join(video_dir, filename[:-4]+'_facecroppad.npy'), np.array(numpy_video))
            if len(numpy_video) != 15:
                print('Error', sess, filename)    
                            
    n_processed += 1      



