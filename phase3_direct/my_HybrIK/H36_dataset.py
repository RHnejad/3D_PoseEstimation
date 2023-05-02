

import numpy as np
from torch.utils.data import Dataset
from utils import camera_parameters, qv_mult
import cv2

systm = "vita17"  #izar,vita17,laptop
act = "Walking"
load_imgs = True
from_videos = False

zero_centre = True
standardize_3d = True
standardize_2d = False
Normalize = True

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

dataset_direcotories = {"izar":"/home/rhossein/venvs/codes/VideoPose3D/data/",
                "vita17":"/data/rh-data/h3.6/npz/", 
                "laptop": "/Users/rh/test_dir/h3.6/VideoPose3D/data/npz/"}  #vita17 used to be: /home/rh/h3.6/dataset/npz/",
data_directory =  dataset_direcotories[systm]
path_positions_2d_VD3d = data_directory + "data_2d_h36m.npz" #"data_2d_h36m_gt.npz" 
path_positions_3d_VD3d =data_directory + "data_3d_h36m.npz"


subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']

KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]
# KeyPoints_from3d = list(range(32))
KeyPoints_from3d_to_delete = [4,5,9,10,11,16,20,21,22,23,24,28,29,30,31]



class H36_dataset(Dataset):
    def __init__(self, num_cams = 1, subjectp=subjects , transform=None, target_transform=None, is_train = True):
        
        self.cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274", ]

        self.dataset2d, self.dataset3d, self.video_and_frame_paths = self.read_data(subjects= subjectp,action=act,is_train = is_train)
        self.dataset2d = self.process_data(self.dataset2d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_2d, z_c = False)
        self.dataset3d = self.process_data(self.dataset3d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, standardize=standardize_3d, z_c = True)

        self.transform = transform
        self.target_transform = target_transform
        self.num_cams = num_cams
        self.is_train = is_train

        self.frame =  np.zeros((1000,1002,3))
        
        


    def __len__(self):
        return len(self.dataset3d) #number of all the frames 

    def __getitem__(self, idx):

        if load_imgs:
            if from_videos:     
                cap = cv2.VideoCapture(self.video_and_frame_paths[idx][0])
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_and_frame_paths[idx][1]) 
                res, self.frame = cap.read()
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            else :
                self.frame = cv2.imread(self.video_and_frame_paths[idx][0])
                # print("***",self.video_and_frame_paths[idx][0], self.frame ,"***")
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            
        keypoints_2d = self.dataset2d[idx].reshape(-1 ,2)

        #resising the image for Resnet
        self.frame = cv2.resize(self.frame, (256, 256))
        self.frame = self.frame/256.0

        return keypoints_2d, self.dataset3d[idx], self.frame #cam 0 

        
    def process_data(self, dataset , sample=sample, is_train = True, standardize = False, z_c = zero_centre) :

        n_frames, n_joints, dim = dataset.shape

        if z_c:
            for i in range(n_frames):
                dataset[i,1:] = dataset[i,1:] - dataset[i,0]


        if is_train :
            data_sum = np.sum(dataset, axis=0)
            data_mean = np.divide(data_sum, n_frames)


            diff_sq2_sum =np.zeros((n_joints,dim))
            for i in range(n_frames):
                diff_sq2_sum += np.power( dataset[i]-data_mean ,2)
            data_std = np.divide(diff_sq2_sum, n_frames)
            data_std = np.sqrt(data_std)

            
            if dim == 2:
                with open("mean_train_2d.npy","wb") as f:
                    np.save(f, data_mean)
                with open("std_train_2d.npy","wb") as f:
                    np.save(f, data_std)  


            elif dim == 3:
                with open("mean_train_3d.npy","wb") as f:
                    np.save(f, data_mean)  
                with open("std_train_3d.npy","wb") as f:
                    np.save(f, data_std)  
                    
                with open("max_train_3d.npy","wb") as f:
                    np.save(f, np.max(dataset, axis=0))  
                with open("min_train_3d.npy","wb") as f:
                    np.save(f, np.min(dataset, axis=0)) 


        if dim == 2:
            with open("mean_train_2d.npy","rb") as f:
                mean_train_2d = np.load(f)
            with open("std_train_2d.npy","rb") as f:
                std_train_2d = np.load(f)  
        elif dim == 3:
            with open("mean_train_3d.npy","rb") as f:
                mean_train_3d =np.load(f)  
            with open("std_train_3d.npy","rb") as f:
                std_train_3d = np.load(f)  
                
            with open("max_train_3d.npy","rb") as f:
                max_train_3d =np.load(f)  
            with open("min_train_3d.npy","rb") as f:
                min_train_3d = np.load(f)  


        if standardize :
            if dim == 2 :
                for i in range(n_frames):
                    if Normalize:
                        # max_dataset, min_dataset = np.max(dataset, axis=0), np.min(dataset, axis=0)
                        # print(max_dataset, min_dataset)
                        # dataset[i] = np.divide(2*dataset[i], (max_dataset-min_dataset))
                        # dataset[i] = dataset[i] - np.divide(min_dataset, (max_dataset-min_dataset))
                        dataset[i] = 2*dataset[i] -1 

                    else:
                        dataset[i] = np.divide(dataset[i] - mean_train_2d, std_train_2d)
            elif dim == 3:
                for i in range(n_frames):
                    if Normalize:
                        # max_dataset, min_dataset = np.max(dataset, axis=0), np.min(dataset, axis=0)
                        dataset[i] = np.divide(dataset[i]- min_train_3d, (max_train_3d-min_train_3d))
                    else:
                        dataset[i] = np.divide(dataset[i] - mean_train_3d, std_train_3d)


        if num_of_joints == 16: #Should through an error if num of joints is 16 but zero centre is false
        
            dataset = dataset[:, 1:, :].copy()
        elif z_c :
            dataset [:,:1,:] *= 0


        if dim == 2 and sample :
            dataset = dataset.reshape((int(n_frames/4),4, num_of_joints,2))

        dataset = dataset[Samples] if sample else dataset

        if dim == 2 and sample :
            dataset = dataset.reshape(-1, num_of_joints,2)  

        return dataset
    
    
    def read_data(self,subjects = subjects, action = "", is_train=True):

        data_file_3d = np.load(path_positions_3d_VD3d, allow_pickle=True)
        data_file_2d = np.load(path_positions_2d_VD3d, allow_pickle=True)

        data_file_3d = data_file_3d['positions_3d'].item()
        data_file_2d = data_file_2d['positions_2d'].item()

        n_frame = 0 
        for s in subjects:
            for a in data_file_3d[s].keys():
                if (action in a ) :
                    n_frame += len(data_file_3d[s][a])  
            

        all_in_one_dataset_3d = np.zeros((4*n_frame if AllCameras else n_frame, 17 ,3),  dtype=np.float32)
        all_in_one_dataset_2d = np.zeros((4*n_frame if AllCameras else n_frame, 17 ,2),  dtype=np.float32)
        video_and_frame_paths = []
        i = 0
        for s in subjects:
            for a in data_file_3d[s].keys():
                if action in a :
                    print(s,a,len(data_file_3d[s][a]))
                    for frame_num in range(len(data_file_3d[s][a])):

                        global_pose = data_file_3d[s][a][frame_num]  
                        global_pose = global_pose[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

                        for c in range(1+3*int(AllCameras)) :

                            tmp = global_pose.copy()

                            if CameraView:
                                for j in range(len(tmp)): 
                                    tmp[j] = tmp[j] - np.divide(np.array(camera_parameters[s][c]['translation']),1000)
                                    tmp[j] = qv_mult(np.array(camera_parameters[s][c]['orientation']),tmp[j])
                                        
                            all_in_one_dataset_3d[i] = tmp

                            tmp2 = data_file_2d[s][a+self.cam_ids[c]][frame_num]
                            
                            all_in_one_dataset_2d[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

                            if load_imgs:
                                if from_videos:
                                    video_and_frame_paths.append( ["/data/rh-data/h3.6/videos/"+s+"/Videos/"+a+self.cam_ids[c]+".mp4",frame_num])
                                else:
                                    if systm == "laptop":
                                        video_and_frame_paths.append( ["/Users/rh/test_dir/h3.6/dataset/S1_frames/"+a+self.cam_ids[c]+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])
                                        # print(video_and_frame_paths)
                                    else:
                                        video_and_frame_paths.append( ["/data/rh-data/h3.6/videos/"+s+"/outputVideos/"+a+self.cam_ids[c]+".mp4/"+str(frame_num+1).zfill(4)+".jpg",frame_num])

                            i = i + 1 

        
        return all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths
