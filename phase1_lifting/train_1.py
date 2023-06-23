import numpy as np
import torch
import torchvision.transforms as T
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
WandB = False


import sys
sys.path.append("../phase3_direct/my_HybrIK/")
from utils import visualize_3d, plot_losses, flip_pose, visualize_2d
from H36_dataset import *

from baselineModel import LinearModel, MyViT

def loss_MPJPE(prediction, target):
    B,J,d =  target.shape
    position_error = torch.norm(prediction - target, dim=-1) #len(target.shape)-1
    metric = torch.sum(position_error, dim=0)
    return metric

def load_statisctics(file_name):
    with open("./logs/run_time_utils/"+file_name+".npy","rb") as f:
        array =np.load(f)
    return array 


def train(batch_size,n_epochs,lr,device,run_name,resume=False):
    
    #Creating Models
    # model_lift = LinearModel(i_dim=num_of_joints*input_dimension, o_dim=num_of_joints*output_dimension,p_dropout=0.5, linear_size=1024, BN=True).to(device)
    model_lift = MyViT().to(device)
    # loss_function = torch.nn.L1Loss()
    loss_function = torch.nn.MSELoss(reduction = "mean")
        
    optimizer_lift = torch.optim.AdamW(model_lift.parameters(),lr = lr)
    
    lr_schdlr_lift = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lift, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )

    if resume:
        model_lift.load_state_dict(torch.load("./logs/models/"+run_name)["model"])
        batch_size = torch.load("./logs/models/"+run_name)["batch_size"]
        last_epoch = torch.load("./logs/models/"+run_name)["epoch"]
        
    training_set = H36_dataset(subjectp=subjects[0:5], is_train = True, action="Posing")#, split_rate=64) #new
    test_set     = H36_dataset(subjectp=subjects[5:7] , is_train = False, action="Posing")#, split_rate=64)
    
    train_loader = DataLoader( training_set, shuffle=True, batch_size=batch_size, num_workers= 1, prefetch_factor=2)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=1, prefetch_factor=2)
   
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

        model_lift.train()       
            
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=True, position=0):

            optimizer_lift.zero_grad()

            y1, y2, frame, _ ,_ = batch
            current_batch_size = y1.shape[0]
            
            y1,y2=y1.float(),y2.float()
            y1, y2 = y1.to(device), y2.to(device) 
            
            #addign noise to input
            # y1 = y1 + (0.05)*torch.randn(current_batch_size, 17, 2).to(device)
 
            y2_hat = model_lift(y1)
            y2_hat = y2_hat.reshape(current_batch_size,num_of_joints,3)

            if Flip:                 
                y1 = flip_pose(y1)
                y2_hat = (flip_pose(model_lift(y1).reshape(current_batch_size,num_of_joints,3))+lift_2d_gt)/2
                y1 = flip_pose(y1)

            loss = loss_function(y2_hat, y2) 
            loss.backward()
            optimizer_lift.step()
            
            train_loss += loss.cpu().item() / len(train_loader)
                
            train_metric_3d += loss_MPJPE(y2_hat, y2)/ len(training_set)
            
        train_metric_3d = torch.mean(train_metric_3d[1:17]) #Please be carefull that here we will have zero for the first joint error so maybe it shoudl be the mean over 1:
        if num_of_joints==17 and zero_centre:
                train_metric_3d *= (17/16)*1000

        lr_schdlr_lift.step(loss) 

        epoch_losses.append(train_loss)
        epoch_metric.append(train_metric_3d.cpu().item())
        
        #________________validation_______________________  
        with torch.no_grad():

            model_lift.eval()          
            val_loss = 0.0
            val_2d_loss = 0.0
            val_metric_3d = torch.zeros(num_of_joints).to(device)
            
            for y1_v, y2_v,frame_v ,_ ,_ in test_loader:
                
                current_batch_size = y1_v.shape[0]
                
                y1_v,y2_v=y1_v.float(),y2_v.float()
                y1_v, y2_v = y1_v.to(device), y2_v.to(device)

                y2_hat_v = model_lift(y1_v).reshape(current_batch_size,17,3)
                    
                if Flip:
                    #flip
                    
                    y2_hat_v = (flip_pose(model_lift(y1_v).reshape(current_batch_size,17,3)) + y2_hat_v) /2
                        
                    #flip back
                    y1_v = flip_pose(y1_v)
                    
                
                loss_v_ = loss_function(y2_hat_v, y2_v)  
                metric_v_3d = loss_MPJPE(y2_hat_v, y2_v) 
            
                val_loss += loss_v_.cpu().item() / len(test_loader)
                val_metric_3d += (metric_v_3d / len(test_set))
            
        val_metric_3d = torch.mean(val_metric_3d[1:17])
        if num_of_joints==17 and zero_centre:
            val_metric_3d *= (17/16)*1000
        
        epoch_val_loss.append(val_loss)
        epoch_val_metric.append(val_metric_3d.cpu().item() )
        
        if WandB:             
            wandb.log({"loss(train)": train_loss, "loss(val.)": val_loss,"MPJPE(train)":train_metric_3d.cpu().item() , " MPJPE(val.)":val_metric_3d.cpu().item()})   
        
        
        print(f"epoch {epoch+1}/{n_epochs} loss(train): {train_loss:.4f} , MPJPE(train):{train_metric_3d.cpu().item()}, loss(val.): {val_loss}, MPJPE(val.){val_metric_3d.cpu().item()}") 
        
    
    #___visualize__train___
        
    y2 = y2.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y2_hat = y2_hat.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y2[0].copy(),y2_hat[0].copy(),   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_a.png")
    visualize_3d(y2[-1].copy(),y2_hat[-1].copy(), "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_b.png")

    y1 = y1.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    y1_hat = np.zeros((1,17,2))#y1_hat.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    frame = frame.cpu().detach().numpy()
    visualize_2d(y1[0].copy(),y1_hat[0].copy(),frame[0].copy(),   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_"+str("train")+"_a.png")
    visualize_2d(y1[-1].copy(),y1_hat[-1].copy(),frame[-1].copy(), "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_"+str("train")+"_b.png")         
        

    plot_losses(epoch_losses,epoch_val_loss,epoch_metric,epoch_val_metric,"./logs/visualizations/"+(resume*"resumed_")+run_name)
    
    #___visualize__validation___
    
    y2_v = y2_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y2_hat_v = y2_hat_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y2_v[0],y2_hat_v[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_test_a.png")
    visualize_3d(y2_v[-1],y2_hat_v[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_test_b.png")
    
    y1_v = y1_v.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    y1_hat_v = np.zeros((1,17,2))#y1_hat.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    frame_v = frame_v.cpu().detach().numpy()
    visualize_2d(y1_v[0].copy(),y1_hat_v[0].copy(),frame_v[0].copy(),   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_"+str("test")+"_a.png")
    visualize_2d(y1_v[-1].copy(),y1_hat_v[-1].copy(),frame_v[-1].copy(), "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_"+str("test")+"_b.png")         
        
    torch.save({'epoch' : epoch, 'batch_size':batch_size, 'model' : model_lift.state_dict(), 'optimizer': optimizer_lift.state_dict()  },"./logs/models/"+(resume*"resumed_")+run_name)
    
    return model_lift

        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",device)
    batch_size = 64
    n_epochs= 150
    lr = 0.0001
    run_name = "june_23_vit_posingall_17_mse"
    CtlCSave = False
    Resume = False
    Train = True
    
    Flip = False
      
    if Train :
        print("___"+run_name+"___")
        if not os.path.exists("./logs/visualizations/"+run_name):
            os.mkdir(os.path.join("./logs/visualizations/", run_name))
            
        if WandB:
            wandb.init(project="loop",name=run_name, config={"learning_rate": lr, "architecture": "CNN","dataset": "H3.6","epochs": n_epochs,})
        
        # try:
        model = train(batch_size,n_epochs,lr,device,run_name,resume=Resume)
        # torch.save(model.state_dict(),"./logs/models/second_"+run_name)
        # except KeyboardInterrupt:
        #         if CtlCSave: torch.save(model.state_dict(),"./logs/models/interrupt_"+run_name)
        
        if WandB:
            wandb.finish() 
            
        print("___"+run_name+" DONE___")
