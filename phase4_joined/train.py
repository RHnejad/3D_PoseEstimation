from Model import Model_3D
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from Custom_Video_dataset import *
from tqdm import tqdm
import os
from utils import visualize_3d, plot_losses
import wandb
from args import *

import sys
sys.path.append("../phase3_direct/my_HybrIK/")
from H36_dataset import *
    
Wandb = 0

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
    
    #Creating the Model
    model_direct= Model_3D().to(device)
    
    loss_function = torch.nn.MSELoss(reduction = "mean")
    optimizer = torch.optim.Adam(model_direct.parameters(),lr = lr)
    
    lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )
    
    if resume:
        model_direct.load_state_dict(torch.load("./logs/"+run_name)["model"])
        batch_size = torch.load("./logs/"+run_name)["batch_size"]
        last_epoch = torch.load("./logs/"+run_name)["epoch"]
        

    # training_set = Custom_video_dataset() 
    # test_set     = Custom_video_dataset()
    
    training_set = H36_dataset(num_cams=num_cameras, subjectp=["S1"], is_train = True) 
    test_set     = H36_dataset(num_cams=num_cameras, subjectp=["S11"] , is_train = False)
    
    train_loader = DataLoader( training_set, shuffle=True, batch_size=batch_size, num_workers= 1)
    val_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=1)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=1)
   
    
    epoch_losses, epoch_metric = list(), list()
    epoch_val_loss, epoch_val_metric  = list(), list()
    
    for epoch in tqdm(range(n_epochs),desc="Training"):
        train_loss = 0.0
        train_metric = torch.zeros(num_of_joints).to(device)

        model_direct.train()  
            
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=False):
            
            optimizer.zero_grad()

            x, y, frame  = batch
     
            x,y=x.float(),y.float()
            x, y = x.to(device), y.to(device) 
            frame = frame.float()
            frame =frame.to(device)
            
            y_hat = model_direct(frame)       
            y_hat = y_hat.reshape(-1,num_of_joints,output_dimension)
            
            loss = loss_function(y_hat, y) 
            
            loss.backward()
            optimizer.step()
              
            train_loss += loss.cpu().item() / len(train_loader)
            train_metric += loss_MPJPE(y_hat, y)/ len(training_set)
            
            
        train_metric = torch.mean(train_metric) #Please be carefull that here we will have zero for the first joint error so maybe it shoudl be the mean over 1:
        if num_of_joints==17 and zero_centre:
                train_metric *= (17/16)*1000
                
            
        # lr_schdlr.step(loss)
        
        epoch_losses.append(train_loss)
        epoch_metric.append(train_metric.cpu().item() )
        
        
        #________________validation_______________________
        with torch.no_grad():
            model_direct.eval()
            val_loss = 0.0
            val_metric = torch.zeros(num_of_joints).to(device)
            
            for x_v, y_v, frame_v  in val_loader:
                
                x_v,y_v=x_v.float(),y_v.float()
                x_v, y_v = x_v.to(device), y_v.to(device)
                
                frame_v = frame_v.float()
                frame_v =frame_v.to(device)
                
                y_hat_v = model_direct(frame_v)
                    
                y_hat_v = y_hat_v.reshape(-1,num_of_joints,output_dimension)
                
                loss_v = loss_function(y_hat_v, y_v) 

                metric_v = loss_MPJPE(y_hat_v, y_v) 
            
                val_loss += loss_v.cpu().item() / len(val_loader)
                val_metric += (metric_v / len(test_set))
            
        val_metric = torch.mean(val_metric)
        if num_of_joints==17 and zero_centre:
            val_metric *= (17/16)*1000
        
        epoch_val_loss.append(val_loss)
        epoch_val_metric.append(val_metric.cpu().item() )
                
        if Wandb:             
            wandb.log({"loss(train)": train_loss, "loss(val.)": val_loss,"MPJPE(train)":train_metric.cpu().item() , " MPJPE(val.)":val_metric.cpu().item()})   
           
        print(f"epoch {epoch+1}/{n_epochs} loss(train): {train_loss:.4f} , MPJPE(train):{train_metric.cpu().item()}, loss(val.): {val_loss}, MPJPE(val.){val_metric.cpu().item()}") 
        
    y = y.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y_hat = y_hat.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y[0],y_hat[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_a.png")
    visualize_3d(y[-1],y_hat[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_b.png")     
    
    plot_losses(epoch_losses,epoch_val_loss,epoch_metric,epoch_val_metric,"./logs/visualizations/"+(resume*"resumed_")+run_name)
    
    y_v = y_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y_hat_v = y_hat_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y_v[0],y_hat_v[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_test_a.png")
    visualize_3d(y_v[-1],y_hat_v[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_test_b.png")         
    
    torch.save({'epoch' : epoch, 'batch_size':batch_size, 'model' : model_direct.state_dict(), 'optimizer': optimizer.state_dict() , 'scheduler': lr_schdlr.state_dict() },"./logs/"+(resume*"resumed_")+run_name)
    
    #____test_____
    predicted_poses = []
    gt_poses = []
    with torch.no_grad():
        model_direct.eval()
        
        for x_t, y_t, frame_t  in test_loader:
            
            x_t,y_t=x_t.float(),y_t.float()
            x_t, y_t = x_t.to(device), y_t.to(device)
            
            frame_t = frame_t.float()
            frame_t =frame_t.to(device)
            
            y_hat_t = model_direct(frame_t)
                
            y_hat_t = y_hat_t.reshape(-1,num_of_joints,output_dimension)
            
            predicted_poses.append(y_hat_t.cpu().detach().numpy().flatten())
            gt_poses.append(y_t.cpu().detach().numpy().flatten())
            
    # predicted_poses = np.array(predicted_poses).reshape((-1,17,3))
    # gt_poses = np.array(gt_poses).reshape((-1,17,3))
    # for i in range(predicted_poses.shape[0]):
    #     visualize_3d(gt_poses[i].copy(), predicted_poses[i].copy(), "./logs/visualizations/"+str(run_name)+"/"+str(i).zfill(4)+".png")
  
    return model_direct



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",device)
    batch_size = 8
    n_epochs= 10
    lr = 0.001 #0.001
    run_name = "yoga_vita17"
    CtlCSave = False
    Resume = True
    Train = True
    
    if Train :
        if not os.path.exists("./logs/visualizations/"+(Resume*"resumed_")+run_name):
            os.mkdir(os.path.join("./logs/visualizations/", (Resume*"resumed_")+run_name))
            
        if Wandb:
            wandb.init(project="Phase4",name=run_name, config={"learning_rate": lr, "architecture": "CNN","dataset": "Custom","epochs": n_epochs,})
        
        try:
            model = train(batch_size,n_epochs,lr,device,run_name,resume=Resume)
            torch.save(model.state_dict(),"./logs/second_"+run_name)
        except KeyboardInterrupt:
                if CtlCSave: torch.save(model.state_dict(),"./logs/"+run_name)
        
        if Wandb:
            wandb.finish()
