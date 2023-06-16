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

from Model_2d import Model_2D, Projection
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
    model_2d= Model_2D().to(device)
    model_3d= Model_3D().to(device)
    # model_lift = LinearModel(i_dim=num_of_joints*input_dimension, o_dim=num_of_joints*output_dimension,p_dropout=0.5, linear_size=1024).to(device)
    model_lift = MyViT().to(device)
    if Project:
        model_proj = Projection().to(device)
    
    if Triangle:
        loss_function = TriangleLoss(Project)
    else:
        loss_function = torch.nn.L1Loss()
        # loss_function = torch.nn.MSELoss(reduction = "mean")
        
    
    optimizer_2d = torch.optim.Adam(model_2d.parameters(),lr = lr)#, weight_decay=1e-8 
    optimizer_3d = torch.optim.Adam(model_3d.parameters(),lr = lr)
    optimizer_lift = torch.optim.Adam(model_lift.parameters(),lr = 0.0001)
    if Project:
        optimizer_proj = torch.optim.Adam(model_proj.parameters(),lr = lr)
    
    lr_schdlr_3d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_3d, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )
    lr_schdlr_2d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2d, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )
    lr_schdlr_lift = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lift, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )
    lr_schdlr_proj = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_proj, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )

    if resume:
        model_2d.load_state_dict(torch.load("./logs/models/"+run_name)["model"])
        batch_size = torch.load("./logs/models/"+run_name)["batch_size"]
        last_epoch = torch.load("./logs/models/"+run_name)["epoch"]
        
    training_set = H36_dataset(subjectp=subjects[0:5], is_train = True, action="Walking 1.", split_rate=81) #new
    test_set     = H36_dataset(subjectp=subjects[5:7] , is_train = False, action="Walking 1.", split_rate=64)
    
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

        model_2d.train()
        model_3d.train()
        model_lift.train()
        if Project: model_proj.train()
            
            
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=True, position=0):

            optimizer_3d.zero_grad()
            optimizer_2d.zero_grad()
            optimizer_lift.zero_grad()
            if Project:
                optimizer_proj.zero_grad()

            y1, y2, frame, _  = batch
            current_batch_size = y1.shape[0]
            
            y1,y2=y1.float(),y2.float()
            y1, y2 = y1.to(device), y2.to(device) 
            frame = frame.float()
            frame =frame.to(device)
            frame = torch.permute(frame, (0,3,1,2))
                           
            y1_hat = model_2d(frame).reshape(current_batch_size,num_of_joints,2)   
            y2_hat, heatmap2_hat = model_3d(frame).reshape(current_batch_size,num_of_joints,3)
            if Triangle:
                lift_2d_pred = model_lift(y1_hat).reshape(current_batch_size,num_of_joints,3)
                lift_2d_gt = model_lift(y1).reshape(current_batch_size,num_of_joints,3)
                
                if Project: 
                    proj_3d_pred = model_proj(y2_hat).reshape(current_batch_size,num_of_joints,2)
                    proj_3d_gt = model_proj(y2).reshape(current_batch_size,num_of_joints,2)
            
            if Flip:
                               
                frame = torch.flip(frame, (3,))

                y1 = flip_pose(y1)
                
                y1_hat = (flip_pose(model_2d(frame).reshape(current_batch_size,num_of_joints,2))+y1_hat)/2
                y2_hat =  (flip_pose(model_3d(frame).reshape(current_batch_size,num_of_joints,3))+y2_hat)/2
                
                if Triangle:
                    lift_2d_pred = (flip_pose(model_lift(y1_hat).reshape(current_batch_size,num_of_joints,3))+lift_2d_pred)/2
                    lift_2d_gt = (flip_pose(model_lift(y1).reshape(current_batch_size,num_of_joints,3))+lift_2d_gt)/2
                    
                    if Project: 
                        
                        y2 = flip_pose(y2)
                        
                        proj_3d_pred = (flip_pose(model_proj(y2_hat).reshape(current_batch_size,num_of_joints,2))+proj_3d_pred)/2
                        proj_3d_gt =  (flip_pose(model_proj(y2).reshape(current_batch_size,num_of_joints,2))+proj_3d_gt)/2
                        
                        y2 = flip_pose(y2)
 
                #flip back
                frame = torch.flip(frame, (3,))
                y1 = flip_pose(y1)
                    
    
            if Triangle: 
                
                if Project:
                    loss, loss_2d_, loss_3d_, loss_lift_, loss_proj_  = loss_function(predicted_2d = y1_hat, predicted_3d = y2_hat,
                                lift_2d_gt = lift_2d_gt, lift_2d_pred=lift_2d_pred,
                                gt_2d = y1, gt_3d = y2, proj_3d_pred = proj_3d_pred.clone() , proj_3d_gt = proj_3d_gt.clone())
                else:                    
                    #predicted_2d, predicted_3d, lift_2d_gt, lift_2d_pred , gt_2d, gt_3d
                    loss, loss_2d_, loss_3d_, loss_lift_, loss_proj_  = loss_function(predicted_2d = y1_hat, predicted_3d = y2_hat,
                                     lift_2d_gt = lift_2d_gt, lift_2d_pred=lift_2d_pred,
                                     gt_2d = y1, gt_3d = y2)
                loss.backward()  
            
            else :
                loss_2d = loss_function(y1_hat, y1) 
                loss_3d = loss_function(y2_hat, y2) 
                # loss_lift = loss_function(y2_hat_lift, y2_hat) 
    
                loss_2d.backward()
                loss_3d.backward()


            optimizer_2d.step()
            optimizer_3d.step()
            if Triangle:
                optimizer_lift.step()
                if Project:
                    optimizer_proj.step()
            
            if Triangle:
                train_loss += loss.cpu().item() / len(train_loader)
            else:
                train_loss += loss_2d.cpu().item() / len(train_loader)
                
            train_metric_3d += loss_MPJPE(y2_hat, y2)/ len(training_set)
            
        train_metric_3d = torch.mean(train_metric_3d[1:17]) #Please be carefull that here we will have zero for the first joint error so maybe it shoudl be the mean over 1:
        if num_of_joints==17 and zero_centre:
                train_metric_3d *= (17/16)*1000
                

        if Triangle:
            lr_schdlr_3d.step(loss_3d_) #fix this for both 2d and 3d
            lr_schdlr_2d.step(loss_2d_) 
            lr_schdlr_lift.step(loss_lift_) 
            if Project:
                lr_schdlr_proj.step(loss_proj_) 
        else:
            lr_schdlr_3d.step(loss_3d) #fix this for both 2d and 3d
            lr_schdlr_2d.step(loss_2d) 
        
        
        epoch_losses.append(train_loss)
        epoch_metric.append(train_metric_3d.cpu().item())
        
        #________________validation_______________________  
        with torch.no_grad():

            model_2d.eval()
            model_3d.eval()
            model_lift.eval()
            if Project:
                model_proj.eval()
            
            val_loss = 0.0
            val_2d_loss = 0.0
            val_metric_3d = torch.zeros(num_of_joints).to(device)
            
            for y1_v, y2_v, frame_v, _  in test_loader:
                
                current_batch_size = y1_v.shape[0]
                
                y1_v,y2_v=y1_v.float(),y2_v.float()
                y1_v, y2_v = y1_v.to(device), y2_v.to(device)
                
                frame_v = frame_v.float()
                frame_v =frame_v.to(device)
                frame_v = torch.permute(frame_v, (0,3,1,2))
                
                y1_hat_v = model_2d(frame_v).reshape(current_batch_size,17,2)
                y2_hat_v, heatmap2_hat_v = model_3d(frame_v).reshape(current_batch_size,17,3)

                if Triangle:
                    lift_2d_pred_v = model_lift(y1_hat_v).reshape(current_batch_size,17,3)
                    lift_2d_gt_v = model_lift(y1_v).reshape(current_batch_size,17,3)
                    
                    if Project:
                        proj_3d_pred_v = model_proj(y2_hat_v).reshape(current_batch_size,17,2)
                        proj_3d_gt_v = model_proj(y2_v).reshape(current_batch_size,17,2)
                    
                    
                if Flip:
                    #flip
                    frame_v = torch.flip(frame_v, (3,))
                    y1_v = flip_pose(y1_v)
                    
                    y1_hat_v = (flip_pose(model_2d(frame_v).reshape(current_batch_size,17,2)) + y1_hat_v) /2
                    y2_hat_v = (flip_pose(model_3d(frame_v).reshape(current_batch_size,17,3)) + y2_hat_v) /2
                    
                    if Triangle:
                        lift_2d_pred_v = (flip_pose(model_lift(y1_hat_v).reshape(current_batch_size,17,3)) + lift_2d_pred_v) /2
                        lift_2d_gt_v = (flip_pose(model_lift(y1_v).reshape(current_batch_size,17,3)) + lift_2d_gt_v) /2
                        
                        if Project:
                            
                            y2_v = flip_pose(y2_v)
                            
                            proj_3d_pred_v = (flip_pose(model_proj(y2_hat_v).reshape(current_batch_size,17,2)) + proj_3d_pred_v  )/2
                            proj_3d_gt_v = (flip_pose(model_proj(y2_v).reshape(current_batch_size,17,2)) +  proj_3d_gt_v  )/2
                            
                            #flip back
                            y2_v = flip_pose(y2_v)
                                
    
                    #flip back
                    frame_v = torch.flip(frame_v, (3,))
                    y1_v = flip_pose(y1_v)
                    
                
                if Triangle:
                    if Project:
                        loss_v, loss_2d_v_, loss_3d_v_, loss_lift_v_, loss_proj_v_  = loss_function(predicted_2d = y1_hat_v, predicted_3d = y2_hat_v,
                                        lift_2d_gt = lift_2d_gt_v , lift_2d_pred = lift_2d_pred_v,
                                        gt_2d = y1_v, gt_3d = y2_v,  proj_3d_pred = proj_3d_pred_v.clone() , proj_3d_gt = proj_3d_gt_v.clone() )
                        
                    else:
                        loss_v, loss_2d_v_, loss_3d_v_, loss_lift_v_, loss_proj_v_   = loss_function(predicted_2d = y1_hat_v, predicted_3d = y2_hat_v,
                                        lift_2d_gt = lift_2d_gt_v , lift_2d_pred = lift_2d_pred_v,
                                        gt_2d = y1_v, gt_3d = y2_v)
                else :
                    loss_2d_v_ = loss_function(y1_hat_v, y1_v) 
                    
                metric_v_3d = loss_MPJPE(y2_hat_v, y2_v) 
            
                if Triangle:
                    val_loss += loss_v.cpu().item() / len(test_loader)
                    val_2d_loss += loss_2d_v_.cpu().item() / len(test_loader)
                else:
                    val_loss += loss_2d_v_.cpu().item() / len(test_loader)
                val_metric_3d += (metric_v_3d / len(test_set))
            
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
        

    visualize(y1,y2,y1_hat,y2_hat,lift_2d_pred,proj_3d_pred,frame,run_name, "train", resume) 
    plot_losses(epoch_losses,epoch_val_loss,epoch_metric,epoch_val_metric,"./logs/visualizations/"+(resume*"resumed_")+run_name)
    visualize(y1_v,y2_v,y1_hat_v,y2_hat_v,lift_2d_pred_v,proj_3d_pred_v,frame_v,run_name,"test", resume)
        
    #, 'scheduler': lr_schdlr.state_dict()
    torch.save({'epoch' : epoch, 'batch_size':batch_size, 'model' : model_2d.state_dict(), 'optimizer': optimizer_2d.state_dict()  },"./logs/models/"+(resume*"resumed_")+run_name)
    
    return model_2d, model_3d, model_lift

def custom():
    pass

        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",device)
    batch_size = 32
    n_epochs= 100
    lr = 0.001
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

