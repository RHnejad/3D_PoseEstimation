

import numpy as np
from torch.utils.data import Dataset
from utils import camera_parameters, qv_mult, flip_pose
import cv2
import albumentations as A

systm = "vita17"  #izar,vita17,laptop
act = "Walking" #"Walking"
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
Mono_3d_file= True
if AllCameras:
    CameraView = True
MaxNormConctraint = False 


num_cameras = 1
input_dimension = num_cameras*2
output_dimension = 3

num_of_joints = 17 #data = np.insert(data, 0 , values= [0,0,0], axis=0 )

dataset_direcotories = {"izar":"/work/vita/datasets/h3.6", #/home/rhossein/venvs/codes/VideoPose3D/data/
                "vita17":"/data/rh-data/h3.6", 
                "laptop": "/Users/rh/test_dir/h3.6/VideoPose3D/data"}  #vita17 used to be: /home/rh/h3.6/dataset/npz/",

data_directory =  dataset_direcotories[systm]
path_positions_2d_VD3d = data_directory + "/npz/data_2d_h36m.npz" #"data_2d_h36m_gt.npz" 
path_positions_3d_VD3d =data_directory + "/npz/data_3d_h36m.npz"
path_positions_3d_VD3d_mono =data_directory + "/npz/data_3d_h36m_mono.npz"


subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
# KeyPoints_from3d = list(range(32))
KeyPoints_from3d_to_delete = [4,5,9,10,11,16,20,21,22,23,24,28,29,30,31]


class H36_dataset(Dataset):
    def __init__(self, subjectp=subjects , action=act, transform=None, target_transform=None, is_train = True, split_rate=None):
        
        self.cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]
        
        self.split_rate = split_rate

        self.dataset2d, self.dataset3d, self.video_and_frame_paths = self.read_data(subjects= subjectp,action=action,is_train = is_train)
        
        if self.split_rate:
            self.dataset2d = self.dataset2d[::split_rate]
            self.dataset3d = self.dataset3d[::split_rate]
            self.video_and_frame_paths = self.video_and_frame_paths[::split_rate]
        
        self.dataset2d = self.process_data(self.dataset2d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_2d, z_c = False)
        self.dataset3d = self.process_data(self.dataset3d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_3d, z_c = True)

        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train
        
        self.is_train = is_train
        
    def __len__(self):
        return len(self.dataset3d) #number of all the frames 

    def __getitem__(self, idx):
        pass

if __name__ == "__main__" :
    
    import pickle     
    path = "/home/rh/codes/EpipolarPose/data/h36m/annot/train-fs.pkl"
    
    import sys
    sys.path.append("/home/rh/codes/EpipolarPose/")
     
    with open(path, 'rb') as file:
        # breakpoint()
        data = pickle.load(file)

    print(data[1][0]["image"])
    print(data[2][0]["image"])
    print(data[3][0]["image"])
    print(data[4][0]["image"])
    breakpoint()