import numpy as np 
from torch.utils.data import Dataset
import cv2
import os


systm = "vita17"  #izar,vita17,laptop
act = "" #"Walking"
load_imgs = True
from_videos = False

zero_centre = True
standardize_3d = False
standardize_2d = False
Normalize = False

sample = False
Samples = np.random.randint(0,74872 if act=="Walk" else 389938, 200) #389938+135836=525774
AllCameras = False
CameraView = True 
if AllCameras:
    CameraView = True
MaxNormConctraint = False 

num_cameras = 1
input_dimension = num_cameras*2
output_dimension = 3

num_of_joints = 17 #data = np.insert(data, 0 , values= [0,0,0], axis=0 )


class Custom_video_dataset(Dataset):
    def __init__(self, video = "yoga"):
        
        self.MP_npy_path = "../phase2_opp_mb/MB_npy"    
        self.frames_path = "../phase2_opp_mb/ffmpeg_frames"
        
        self.frames=[]
        self.poses=[]
        
        self.len_dataset = 0
        for file in os.listdir(self.frames_path):
            if video in file and file != ".DS_Store":
                mb_skeleton = np.load(os.path.join(self.MP_npy_path, file+".npy"))
                mb_skeleton = list(mb_skeleton.flatten())
                self.poses.append(mb_skeleton)
                current_video_path = os.path.join(self.frames_path, file)
                for frame in os.listdir(current_video_path):
                    self.len_dataset += 1
                    self.frames.append(os.path.join(current_video_path,frame))
                    
        self.poses = np.array(self.poses).reshape(-1,17,3)
        assert self.poses.shape[0] ==  self.len_dataset
        
        if zero_centre:
            for i in range(self.len_dataset) :
                self.poses[i,1:,:] = self.poses[i,1:,:] - self.poses[i,0,:]            
            self.poses[:,:1,:] *= 0
                        
    def __len__(self):
        return self.len_dataset
        
    def __getitem__(self, idx):
        
        frame = cv2.imread(self.frames[idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        s_0, s_1 =frame.shape[0], frame.shape[1]
        l_min = min(s_0, s_1)
        delta_0 = l_min/2
        delta_1 = l_min/2
    
        frame_ = frame[  int(s_0/2-delta_0) : int(s_0/2+delta_0)  , int(s_1/2-delta_1) : int(s_1/2+delta_1) ,:]

        frame_ = cv2.resize(frame_, (256, 256))
        frame_ = frame_/256.0
        
        return np.zeros((17,2)), self.poses[idx], frame_
    
    
if __name__ == "__main__":

    data = Custom_video_dataset()
    print(data.poses.max())
    print(data.poses.max(axis=0))
    print(data.poses.min())
    print(data.poses.min(axis=0))
    print("****")
    # print(data.__len__())
    print(data.__getitem__(0)[1])
    
    x,y,f = data.__getitem__(-1)
    
    import matplotlib.pyplot as plt
    plt.imshow(f)
    plt.show()