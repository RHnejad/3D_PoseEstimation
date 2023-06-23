#adding noise to training data 
#reducing the complexity of the model 
#

systm = "vita17"  #izar,vita17,laptop

from baselineModel import *

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

import sys
sys.path.append("../phase3_direct/my_HybrIK/") #new
from utils import camera_parameters #new
# from utils import camera_parameters

act = "Walk"
run_num = "test"

load_imgs = True
if load_imgs:
    import cv2

zero_centre = False
standardize_3d = False

standardize_2d = True
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
KeyPoints_from3d_to_delete = [4,5,9,10,11,16,20,21,22,23,24,28,29,30,31]

# global mean_train_2d 
# global mean_train_3d 
# global std_train_2d 
# global std_train_3d 
#NEW (Quaternion Calculations -> source : https://www.meccanismocomplesso.org/en/hamiltons-quaternions-and-3d-rotation-with-python/ )
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def qv_mult(q1, v1):
    # q2 = (0.0,) + v1
    q2 = np.insert(v1,0,0) #new
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]
#END NEW

def read_data(subjects = subjects, action = "", is_train=True):

    cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274" ]


    data_file_3d = np.load(path_positions_3d_VD3d, allow_pickle=True)
    data_file_2d = np.load(path_positions_2d_VD3d, allow_pickle=True)

    data_file_3d = data_file_3d['positions_3d'].item()
    data_file_2d = data_file_2d['positions_2d'].item()

    n_frame = 0
    print(subjects)
    for s in subjects:
        for a in data_file_3d[s].keys():
            if (action in a ) :
                n_frame += len(data_file_3d[s][a]) 
                print(s+a)
                print(len(data_file_3d[s][a]) )
        print("n_frame",n_frame)

    print("n_frame",n_frame)

    all_in_one_dataset_3d = np.zeros((4*n_frame if AllCameras else n_frame, 17 ,3),  dtype=np.float32)
    all_in_one_dataset_2d = np.zeros((4*n_frame if AllCameras else n_frame, 17 ,2),  dtype=np.float32)
    video_and_frame_paths = []
    i = 0
    for s in subjects:
        for a in data_file_3d[s].keys():
            if action in a :
                for frame in range(len(data_file_3d[s][a])):

                    global_pose = data_file_3d[s][a][frame]  
                    global_pose = global_pose[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

                    for c in range(1+3*int(AllCameras)) :

                        tmp = global_pose.copy()

                        if CameraView:
                            for j in range(len(tmp)): 
                                tmp[j] = tmp[j] - np.divide(np.array(camera_parameters[s][c]['translation']),1000)
                                tmp[j] = qv_mult(np.array(camera_parameters[s][c]['orientation']),tmp[j])
                                    
                        all_in_one_dataset_3d[i] = tmp

                        tmp2 = data_file_2d[s][a+cam_ids[c]][frame]
                        all_in_one_dataset_2d[i] = tmp2[ KeyPoints_from3d ,:] #only keeping the 16 or 17 keypoints we want

                        if load_imgs:
                            video_and_frame_paths.append( ["/data/rh-data/h3.6/videos/"+s+"/Videos/"+a+cam_ids[c]+".mp4",frame])

                        i = i + 1 

    
    return all_in_one_dataset_2d, all_in_one_dataset_3d , video_and_frame_paths


#_________

def visualize_3d(keypoints,keypoints2, name="3d"):
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    if keypoints.shape[0] != 17 :
        keypoints = np.insert(keypoints , 0 , values= [0,0,0], axis=0 )
        keypoints2 = np.insert(keypoints2 , 0 , values= [0,0,0], axis=0 )

    plt.figure()
    ax = plt.axes(projection='3d')

    xdata = keypoints.T[0]
    ydata = keypoints.T[1]
    zdata = keypoints.T[2]
    ax.scatter(xdata,ydata,zdata,"b",label="expectations")
    for i in range(17):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], zdata[sk_points[i]] , "b" )

    xdata2 = keypoints2.T[0]
    ydata2 = keypoints2.T[1]
    zdata2 = keypoints2.T[2]
    ax.scatter(xdata2,ydata2,zdata2, "r" , label ="reality")
    for i in range(17):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], zdata2[sk_points[i]] , "r" )

    plt.legend()

    ax.axes.set_xlim3d(left=-2, right=2) 
    ax.axes.set_ylim3d(bottom=-2, top=2) 
    ax.axes.set_zlim3d(bottom=-1 if zero_centre else 0, top=1 if zero_centre else 2) 
    plt.savefig("./imgs/"+name +'.png')
    plt.show()


def visualize(keypoints,st_kp=None, frame=None, name = "kp"):
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
    
    if keypoints.shape[0] != 17 :
        keypoints = np.insert(keypoints , 0 , values= [0,0], axis=0 )
        st_kp = np.insert(st_kp , 0 , values= [(st_kp[0,0]+st_kp[5,0])/2,(st_kp[0,1]+st_kp[5,1])/2], axis=0 )

    plt.figure()

    if load_imgs :
        keypoints[:,0] = keypoints[:,0] *1000
        keypoints[:,1] = keypoints[:,1] *1000
        st_kp[:,0] = st_kp[:,0] *1000
        st_kp[:,1] = st_kp[:,1] *1000
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(keypoints,st_kp )

    plt.plot(keypoints.T[0],keypoints.T[1], "og", markersize=4)
    for i in range(17):
        plt.plot(keypoints.T[0][sk_points[i]], keypoints.T[1][sk_points[i]],  "g" )
    plt.plot(st_kp.T[0],st_kp.T[1], "ob", markersize=2)
    for i in range(17):
        plt.plot(st_kp.T[0][sk_points[i]], st_kp.T[1][sk_points[i]],  "b" )

    plt.xlim([-1,1]), plt.ylim([1,-1])

    plt.savefig("./imgs/" + name +'.png')


def plot_losses(epoch_losses,epoch_eval_loss,epoch_metric,epoch_eval_metric) :

    plt.figure()
    plt.subplot(1, 2, 1)

    plt.plot(epoch_losses)
    plt.plot(epoch_eval_loss)

    plt.xlabel("epoch")
    plt.ylabel("MSE")

    plt.legend(["training","test"])

    plt.subplot(1, 2, 2)
    plt.plot(epoch_metric)
    plt.plot(epoch_eval_metric)

    plt.xlabel("epoch")
    plt.ylabel("MPJPE")

    plt.legend(["training","test"])
    
    plt.savefig("imgs/plot_metric"+str(run_num)+'.pdf')
    plt.show()

#_______

class Pose_KeyPoints(Dataset):
    def __init__(self, num_cams = 1, subjectp=subjects , transform=None, target_transform=None, is_train = True):

        self.dataset2d, self.dataset3d, self.video_and_frame_paths = read_data(subjects= subjectp,action="",is_train = is_train)
        self.dataset2d = self.process_data(self.dataset2d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, action= act,standardize=standardize_2d, z_c = False)
        self.dataset3d = self.process_data(self.dataset3d,  sample = False if len(subjectp)==2 else sample, is_train = is_train, action= act,standardize=standardize_3d, z_c = True)

        self.transform = transform
        self.target_transform = target_transform
        self.num_cams = num_cams
        self.is_train = is_train

        self.frame =  np.zeros((1000,1002,3))

    def __len__(self):
        return len(self.dataset3d) #number of all the frames 

    def __getitem__(self, idx):
        # noise = np.random.normal(0,0.3,num_of_joints*2).reshape(num_of_joints,2)
        # if not self.is_train:
        #     noise = noise * 0

        # noise = noise * 0

        if load_imgs:     
            cap = cv2.VideoCapture(self.video_and_frame_paths[idx][0])
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_and_frame_paths[idx][1]) 
            res, self.frame = cap.read()
            

        return self.dataset2d[idx].reshape(-1 ,2), self.dataset3d[idx], self.frame #cam 0 

        
    def process_data(self, dataset , action = "",sample=sample, is_train = True, standardize = False, z_c = zero_centre) :

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
                # global mean_train_2d 
                # mean_train_2d = data_mean
                # global std_train_2d
                # std_train_2d = data_std
            elif dim == 3:
                with open("mean_train_3d.npy","wb") as f:
                    np.save(f, data_mean)  
                with open("std_train_3d.npy","wb") as f:
                    np.save(f, data_std)  
                # global mean_train_3d 
                # mean_train_3d = data_mean 
                # global std_train_3d
                # std_train_3d = data_std

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


def loss_MPJPE(prediction, target):
    B,J,d =  target.shape
    position_error = torch.norm(prediction - target, dim=-1) #len(target.shape)-1
    metric = torch.sum(position_error, dim=0)
    return metric


def main():

    import time
    t = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    training_set = Pose_KeyPoints(num_cams=num_cameras, subjectp=subjects[0:1], is_train = True) 
    test_set     = Pose_KeyPoints(num_cams=num_cameras, subjectp=subjects[6:7] , is_train = False)

    batch_size=32

    train_loader = DataLoader( training_set, shuffle=True, batch_size=batch_size, num_workers= 4)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=4)

    with open("mean_train_3d.npy","rb") as f:
        mean_train_3d =np.load(f)  
    with open("std_train_3d.npy","rb") as f:
        std_train_3d = np.load(f)  

    mean_k = mean_train_3d [list(range(17-num_of_joints,17)),:]
    std_k = std_train_3d [list(range(17-num_of_joints,17)),:]
    
    #____________________________Defining model and training parameters___________________________
    
    model = MyViT().to(device)
    # model = MyViT((1,num_of_joints*num_cameras,2),n_blocks=2 , hidden_d=8, n_heads=1, out_d=(num_of_joints*output_dimension)).to(device)   #, n_patches=7, n_blocks=2, 
    # model = TransformerAE(2*num_cams, 3, 0.2).to(device)
    # model = AE(input_dimension, output_dimension, n_joints= num_of_joints).to(device)
    # model = LinearModel(i_dim=num_of_joints*input_dimension, o_dim=num_of_joints*output_dimension,p_dropout=0.5, linear_size=1024).to(device)
    model.apply(weight_init)

    # if load_imgs:
        # model_2 = My_CNN(output_dim=output_dimension)

    n_epochs=120
    lr = 0.001

    loss_function = torch.nn.MSELoss(reduction = "mean") #reduction = "sum"
    #loss_function = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(),
                             lr = lr) #, weight_decay = 1e-8

    lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )

    flag = False
    epoch_losses = list()
    epoch_metric = list()
    epoch_eval_loss = list()
    epoch_eval_metric = list()

    for epoch in tqdm(range(n_epochs),desc="Training"):
        train_loss = 0.0
        train_metric = torch.zeros(num_of_joints).to(device)

        model.train()  
            
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=False):
            
            optimizer.zero_grad()

            x, y, frame  = batch


            x,y=x.float(),y.float()
            y = x if (output_dimension == 2) else y

            x, y = x.to(device), y.to(device)

            print("HERE")   

            if load_imgs: 
                frame = frame.float()
                frame.to(device)
                frame = frame.permute((0,3,2,1))
                # t_y = model_2(frame)

            # print("HERE2")           
            # input()

            y_hat = model(x)

            if flag :
                ytmp = y[0].cpu().detach().numpy().reshape(num_of_joints,output_dimension)
                y_hattmp = y_hat[0].cpu().detach().numpy().reshape(num_of_joints,output_dimension) 

                if standardize_3d :
                    ytmp     = np.multiply(ytmp , std_k ) + mean_k
                    y_hattmp = np.multiply(y_hattmp, std_k ) + mean_k

                if output_dimension == 3 :                   
                    visualize_3d(ytmp,y_hattmp,"y3dflag_b") 
                    
                flag = 0

            y_hat = y_hat.reshape(-1,num_of_joints,output_dimension)
            # loss = criterion (y_hat, y) new
            loss = loss_function(y_hat, y) 
            loss.backward()
            
            if MaxNormConctraint:
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

            optimizer.step()


            temp_std = torch.from_numpy(std_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)
            temp_mean = torch.from_numpy(mean_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)


            if standardize_3d :
                y = torch.mul(y , temp_std ) + temp_mean #DeStandardize
                y_hat = torch.mul(y_hat, temp_std ) + temp_mean
            
            train_loss += loss.cpu().item() / len(train_loader)
            train_metric += loss_MPJPE(y_hat, y)/ len(training_set)

    
       
        #________________validation_______________________
        
        with torch.no_grad(): # you should add validation here instead of test 
            model.eval()
            test_loss = 0.0
            test_metric = torch.zeros(num_of_joints).to(device)
            for x_v, y_v, frame_v  in test_loader:

                x_v,y_v = x_v.float(),y_v.float()
                x_v, y_v = x_v.to(device), y_v.to(device)
                if load_imgs: frame_v.to(device)
                y_v = x_v if (output_dimension == 2) else y_v
                
                y_hat_v = model(x_v) 

                y_hat_v = y_hat_v.reshape(-1,num_of_joints,output_dimension)       
                loss_v = loss_function(y_hat_v, y_v) 

                temp_std = torch.from_numpy(std_k).to(device).expand(y_v.shape[0],num_of_joints,output_dimension)
                temp_mean = torch.from_numpy(mean_k).to(device).expand(y_v.shape[0],num_of_joints,output_dimension)

                if standardize_3d :
                    y_v = torch.mul(y_v , temp_std ) + temp_mean
                    y_hat_v = torch.mul(y_hat_v , temp_std ) + temp_mean
            
                metric_v = loss_MPJPE(y_hat_v, y_v)

                test_loss += loss_v.cpu().item() / len(test_loader)
                test_metric += (metric_v / len(test_set))

        lr_schdlr.step(loss_v) #new
        
        test_metric = torch.mean(test_metric)
        train_metric = torch.mean(train_metric)

        epoch_losses.append(train_loss)
        epoch_metric.append(train_metric.cpu().item() )

        epoch_eval_loss.append(test_loss)
        epoch_eval_metric.append(test_metric.cpu().item() )

        print(f"epoch {epoch+1}/{n_epochs} loss(train): {train_loss:.4f} , MPJPE(train):{train_metric.cpu().item()}, loss(val.): {test_loss}, MPJPE(val.){test_metric.cpu().item()}") 
        

    x = x.cpu().detach().numpy().reshape(-1, num_of_joints,int(input_dimension/num_cameras))
    # x2= np.multiply(x[0],training_set.std_2d)+training_set.mean_2d

    y = y.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y_hat = y_hat.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension) 

    if output_dimension == 3 :
        visualize(x[0],x[0].copy(),name = "train_2d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"a")
        visualize(x[-1],x[-1].copy(),name = "train_2d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"b")
        visualize_3d(y[0],y_hat[0],"train_3d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"a")
        visualize_3d(y[-1],y_hat[-1],"train_3d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"b")
          
    print(y.shape, y_hat.shape)


    #________________________Evaluation _______________________
 
    with torch.no_grad():
        model.eval()
        test_loss = 0.0
        test_metric = torch.zeros(num_of_joints).to(device)
        for x, y, frame in test_loader:
            x,y=x.float(),y.float()
            x, y = x.to(device), y.to(device)
            if load_imgs: frame.to(device)
            y = x if (output_dimension == 2) else y
            
            y_hat = model(x) 
 
            y_hat = y_hat.reshape(-1,num_of_joints,output_dimension)

            temp_std = torch.from_numpy(std_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)
            temp_mean = torch.from_numpy(mean_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)

            loss = loss_function(y_hat, y) 

            if standardize_3d :
                y = torch.mul(y , temp_std ) + temp_mean
                y_hat = torch.mul(y_hat , temp_std ) + temp_mean
                   
            metric = loss_MPJPE(y_hat, y)

            test_loss += loss.cpu().item() / len(test_loader)
            test_metric += (metric / len(test_set))


        test_metric = torch.mean(test_metric)    
        print(f">>TEST loss: {test_loss:.2f} , "+str(test_metric.cpu().item())) 

    #Visualizing Skeleton

    x = x.cpu().detach().numpy().reshape(-1, num_of_joints, int(input_dimension/num_cameras))
    # x2= np.multiply(x[0],training_set.std_2d)+training_set.mean_2d

    y = y.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y_hat = y_hat.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)

    
    if output_dimension == 3 :
        visualize(x[0],x[0].copy(),name = "test_2d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"a")
        visualize(x[int(len(y)/2)],x[int(len(y)/2)].copy(),name = "test_2d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"b")
        visualize(x[-1],x[-1].copy(),name = "test_2d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"c")
        visualize_3d(y[0],y_hat[0],"test_3d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"a")
        visualize_3d(y[int(len(y)/2)],y_hat[int(len(y)/2)],"test_3d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"b")
        visualize_3d(y[-1],y_hat[-1],"test_3d_s"+str(sample)+"z_"+str(zero_centre)+str(run_num)+"c")
  
    elapsed = time.time()-t 
    print("time:",elapsed/60)

    with open("losses"+str(run_num)+".txt","w") as f  :
        f.write("train loss: "+str(train_loss )+"  "+str(train_metric.item() *1000)+"\ntest loss: "+str(test_loss)+"  "+str(test_metric.item() *1000)+ "\n"+str(elapsed/60))

    
    plot_losses(epoch_losses,epoch_eval_loss,epoch_metric,epoch_eval_metric)
    


if __name__ == "__main__" :
    print("start")
    main()

    # training_set = Pose_KeyPoints(num_cams=num_cameras, subjectp=subjects[0:1], is_train = True) 
    # d2d, d3d, frame = training_set.__getitem__(34857)
    # cv2.imwrite("./frame0.png", frame )
    # # print(d2d,d3d,"+",img)

    # visualize(d2d, d2d.copy(),frame, "TEST2d")
    # visualize_3d(d3d,d3d, "TEST3d")

