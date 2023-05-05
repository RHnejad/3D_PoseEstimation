from Model import Model_3D
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from H36_dataset import *
from tqdm import tqdm
import os
from utils import visualize_3d, plot_losses
import wandb

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
    training_set = H36_dataset(num_cams=num_cameras, subjectp=subjects[0:5], is_train = True) 
    test_set     = H36_dataset(num_cams=num_cameras, subjectp=subjects[5:7] , is_train = False)
    
    train_loader = DataLoader( training_set, shuffle=True, batch_size=batch_size, num_workers= 1)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=1)

    
    mean_train_3d, std_train_3d = load_statisctics("mean_train_3d"), load_statisctics("std_train_3d")
    max_train_3d, min_train_3d = load_statisctics("max_train_3d"), load_statisctics("min_train_3d")
    max_train_3d, min_train_3d = torch.from_numpy(max_train_3d).to(device), torch.from_numpy(min_train_3d).to(device)
    if zero_centre and num_of_joints==17 : 
        max_train_3d[:1,:] *= 0 
        min_train_3d[:1,:] *= 0

    mean_k = mean_train_3d [list(range(17-num_of_joints,17)),:]
    std_k = std_train_3d [list(range(17-num_of_joints,17)),:]    
    
    #Creating the Model
    model_direct= Model_3D().to(device)
    if resume:
        model_direct.load_state_dict(torch.load("./logs/"+run_name))
        

    loss_function = torch.nn.MSELoss(reduction = "mean")
    optimizer = torch.optim.Adam(model_direct.parameters(),lr = lr)
    
    lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )
    
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
            
            if standardize_3d : 
                if Normalize:
                    y = torch.mul(y , max_train_3d-min_train_3d ) + min_train_3d 
                    y_hat = torch.mul(y_hat,  max_train_3d-min_train_3d ) + min_train_3d 
                else:
                    temp_std = torch.from_numpy(std_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)
                    temp_mean = torch.from_numpy(mean_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)
                    y = torch.mul(y , temp_std ) + temp_mean #DeStandardize
                    y_hat = torch.mul(y_hat, temp_std ) + temp_mean
            
            train_loss += loss.cpu().item() / len(train_loader)
            train_metric += loss_MPJPE(y_hat, y)/ len(training_set)
            
            
        train_metric = torch.mean(train_metric) #Please be carefull that here we will have zero for the first joint error so maybe it shoudl be the mean over 1:
        if num_of_joints==17 and zero_centre:
                train_metric *= (17/16)
                
            
        lr_schdlr.step(loss)
        
        epoch_losses.append(train_loss)
        epoch_metric.append(train_metric.cpu().item() )
        
        
        #________________validation_______________________  
        with torch.no_grad():
            model_direct.eval()
            val_loss = 0.0
            val_metric = torch.zeros(num_of_joints).to(device)
            
            for x_v, y_v, frame_v  in test_loader:
                
                x_v,y_v=x_v.float(),y_v.float()
                x_v, y_v = x_v.to(device), y_v.to(device)
                
                frame_v = frame_v.float()
                frame_v =frame_v.to(device)
                
                y_hat_v = model_direct(frame_v)
                    
                y_hat_v = y_hat_v.reshape(-1,num_of_joints,output_dimension)
                
                loss_v = loss_function(y_hat_v, y_v) 
                
            
                if standardize_3d :
                    if Normalize:
                        y_v = torch.mul(y_v , max_train_3d-min_train_3d ) + min_train_3d 
                        y_hat_v = torch.mul(y_hat_v,  max_train_3d-min_train_3d ) + min_train_3d 
                    else:
                        y_v = torch.mul(y_v , temp_std ) + temp_mean #DeStandardize
                        y_hat_v = torch.mul(y_hat_v, temp_std ) + temp_mean   
                    
                    
                metric_v = loss_MPJPE(y_hat_v, y_v) 
            
                val_loss += loss_v.cpu().item() / len(test_loader)
                val_metric += (metric_v / len(test_set))
            
        val_metric = torch.mean(val_metric)
        if num_of_joints==17 and zero_centre:
            val_metric *= (17/16)
        
        epoch_val_loss.append(val_loss)
        epoch_val_metric.append(val_metric.cpu().item() )
                     
        wandb.log({"loss(train)": train_loss, "loss(val.)": val_loss,"MPJPE(train)":train_metric.cpu().item() , " MPJPE(val.)":val_metric.cpu().item()})   
           
        print(f"epoch {epoch+1}/{n_epochs} loss(train): {train_loss:.4f} , MPJPE(train):{train_metric.cpu().item()}, loss(val.): {val_loss}, MPJPE(val.){val_metric.cpu().item()}") 
        
    y = y.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y_hat = y_hat.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y[0],y_hat[0],   "./logs/visualizations/"+str(run_name)+"/3d_train_a.png")
    visualize_3d(y[-1],y_hat[-1], "./logs/visualizations/"+str(run_name)+"/3d_train_b.png")     
    
    plot_losses(epoch_losses,epoch_val_loss,epoch_metric,epoch_val_metric,"./logs/visualizations/"+run_name)
    
    return model_direct

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",device)
    batch_size = 64
    n_epochs= 10
    lr = 0.005 #0.001
    run_name = "test_s1_walk 1_may5"
    CtlCSave = False
    
    
    if not os.path.exists("./logs/visualizations/"+run_name):
        os.mkdir(os.path.join("./logs/visualizations/", run_name))
        
    wandb.init(project="Direct_3D_Pose",name=run_name, config={"learning_rate": lr, "architecture": "CNN","dataset": "H3.6","epochs": 30,})
    
    try:
        model = train(batch_size,n_epochs,lr,device,run_name,resume=True)
        torch.save(model.state_dict(),"./logs/"+run_name)
    except KeyboardInterrupt:
            if CtlCSave: torch.save(model.state_dict(),"./logs/"+run_name)
    
    wandb.finish()
    