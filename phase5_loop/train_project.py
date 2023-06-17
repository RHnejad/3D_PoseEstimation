import numpy as np
import torch
import torchvision.transforms as T
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
WandB = False

# ln -s /data2/rh-data/h3.6/Images   /home/rh/codes/EpipolarPose/data/h36m/images

from Model_2d import  Projection
from losses import loss_MPJPE, TriangleLoss, TriangleLoss_sep    
from visualize  import visualize

import sys
sys.path.append("../phase3_direct/my_HybrIK/")
from utils import visualize_3d,visualize_2d, plot_losses, flip_pose
from Model import Model_3D
from H36_dataset import *

sys.path.append("../phase1_lifting/")
from baselineModel import LinearModel, MyViT


def load_statisctics(file_name):
    with open("./logs/run_time_utils/"+file_name+".npy","rb") as f:
        array =np.load(f)
    return array 


def train(batch_size,n_epochs,lr,device,run_name,resume=False, Triangle=True, Flip=False, Project = False):
    
    #Creating Models
    # model_proj = Projection().to(device)
    model_proj = MyViT(chw=(1,17,3), out_d=2).to(device)
    
    loss_function = torch.nn.L1Loss()
    # loss_function = torch.nn.MSELoss(reduction = "mean")
 
    optimizer_proj = torch.optim.Adam(model_proj.parameters(),lr = lr)
    
    lr_schdlr_proj = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_proj, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )

    if resume:
        model_proj.load_state_dict(torch.load("./logs/models/"+run_name)["model"])
        batch_size = torch.load("./logs/models/"+run_name)["batch_size"]
        last_epoch = torch.load("./logs/models/"+run_name)["epoch"]
        
    training_set = H36_dataset(subjectp=subjects[0:5], is_train = True, action="", split_rate=81) #new
    test_set     = H36_dataset(subjectp=subjects[5:7] , is_train = False, action="", split_rate=64)
    
    train_loader = DataLoader( training_set, shuffle=True, batch_size=batch_size, num_workers= 2, prefetch_factor=2)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=2, prefetch_factor=2)
   
    mean_train_3d, std_train_3d = load_statisctics("mean_train_3d"), load_statisctics("std_train_3d")
    max_train_3d, min_train_3d = load_statisctics("max_train_3d"), load_statisctics("min_train_3d")
    max_train_3d, min_train_3d = torch.from_numpy(max_train_3d).to(device), torch.from_numpy(min_train_3d).to(device)
    if zero_centre and num_of_joints==17 : 
        max_train_3d[:1,:] *= 0 
        min_train_3d[:1,:] *= 0

    mean_k = mean_train_3d [list(range(17-num_of_joints,17)),:]
    std_k = std_train_3d [list(range(17-num_of_joints,17)),:]    
    
    epoch_losses, epoch_metric = list(), list()
    epoch_val_loss, epoch_val_metric  = list(), list()
    
    for epoch in tqdm(range(n_epochs),desc="Training"):
        train_loss = 0.0
        train_metric_3d = torch.zeros(num_of_joints).to(device)

        model_proj.train()
            
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=True, position=0):

            optimizer_proj.zero_grad()

            y1, y2, frame, _  = batch
            current_batch_size = y1.shape[0]
            
            y1,y2 =y1.float(),y2.float()
            y1, y2 = y1.to(device), y2.to(device) 
            frame = frame.float()
            frame =frame.to(device)
            frame = torch.permute(frame, (0,3,1,2))
            
                           
            y1_hat = model_proj(y2).reshape(current_batch_size,num_of_joints,2)
    
            loss_3d = loss_function(y1_hat, y1) 
    
            loss_3d.backward()

            optimizer_proj.step()
            
            train_loss += loss_3d.cpu().item() / len(train_loader)
                            
        train_metric_3d = torch.mean(train_metric_3d[1:17]) #Please be carefull that here we will have zero for the first joint error so maybe it shoudl be the mean over 1:
        if num_of_joints==17 and zero_centre:
                train_metric_3d *= (17/16)*1000
                

        lr_schdlr_proj.step(loss_3d) 
        
        epoch_losses.append(train_loss)
        epoch_metric.append(train_metric_3d.cpu().item())
        
        #________________validation_______________________  
        with torch.no_grad():

            model_proj.eval()
            
            val_loss = 0.0
            val_2d_loss = 0.0
            val_metric_3d = torch.zeros(num_of_joints).to(device)
            
            for y1_v, y2_v,frame_v, _  in test_loader:
                
                current_batch_size = y1_v.shape[0]
                
                y1_v,y2_v=y1_v.float(),y2_v.float()
                y1_v, y2_v = y1_v.to(device), y2_v.to(device)
                frame_v = frame_v.float()
                frame_v =frame_v.to(device)
                frame_v = torch.permute(frame_v, (0,3,1,2))
                
                y1_hat_v = model_proj(y2_v).reshape(current_batch_size,17,2)
                          
                loss_2d_v_ = loss_function(y1_hat_v, y1_v) 
                    
                val_loss += loss_2d_v_.cpu().item() / len(test_loader)
            
        val_metric_3d = torch.mean(val_metric_3d[1:17])
        if num_of_joints==17 and zero_centre:
            val_metric_3d *= (17/16)*1000
        
        epoch_val_loss.append(val_loss)
        epoch_val_metric.append(val_metric_3d.cpu().item() )
        
        if WandB:             
            wandb.log({"loss(train)": train_loss, "loss(val.)": val_loss,"MPJPE(train)":train_metric_3d.cpu().item() , " MPJPE(val.)":val_metric_3d.cpu().item()})   
        
        if Triangle:
            print("___losses___")   
            loss_function.report_losses()
            print("val_2d_loss:", val_2d_loss)
 
        print(f"epoch {epoch+1}/{n_epochs} loss(train): {train_loss:.4f} , MPJPE(train):{train_metric_3d.cpu().item()}, loss(val.): {val_loss}, MPJPE(val.){val_metric_3d.cpu().item()}") 
        

    visualize(y1,y2,y1_hat,[],[],[],frame,run_name, "train", resume) 
    plot_losses(epoch_losses,epoch_val_loss,epoch_metric,epoch_val_metric,"./logs/visualizations/"+(resume*"resumed_")+run_name)
    visualize(y1_v,y2_v,y1_hat_v,[],[],[],frame_v,run_name,"test", resume)
        
    #, 'scheduler': lr_schdlr.state_dict()
    torch.save({'epoch' : epoch, 'batch_size':batch_size, 'model' : model_proj.state_dict(), 'optimizer': optimizer_proj.state_dict()  },"./logs/models/"+(resume*"resumed_")+run_name)
    
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",device)
    batch_size = 64
    n_epochs= 100
    lr = 0.0001
    run_name = "june_16_tr_pr"
    CtlCSave = False
    Resume = False
    Train = True
    
    Triangle = 0
    Flip = 0
    Project = 1
    
    if Train :
        print("___"+run_name+"___")
        if not os.path.exists("./logs/visualizations/"+run_name):
            os.mkdir(os.path.join("./logs/visualizations/", run_name))
            
        if WandB:
            wandb.init(project="loop",name=run_name, config={"learning_rate": lr, "architecture": "CNN","dataset": "H3.6","epochs": n_epochs,})
        
        # try:
        model = train(batch_size,n_epochs,lr,device,run_name,resume=Resume, Triangle=Triangle, Flip=Flip, Project = Project)
        # torch.save(model.state_dict(),"./logs/models/second_"+run_name)
        # except KeyboardInterrupt:
        #         if CtlCSave: torch.save(model.state_dict(),"./logs/models/interrupt_"+run_name)
        
        if WandB:
            wandb.finish() 
            
        print("___"+run_name+" DONE___")

