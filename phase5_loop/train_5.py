import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
WandB = True

from Model_2d import Model_2D

import sys
sys.path.append("../phase3_direct/my_HybrIK/")
from utils import visualize_3d,visualize_2d, plot_losses
from H36_dataset import *

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
    # model_direct_3d= Model_3D().to(device)
    model_2d= Model_2D().to(device)
    
    loss_function = torch.nn.MSELoss(reduction = "mean")
    # optimizer = torch.optim.Adam(model_direct.parameters(),lr = lr)
    optimizer_2d = torch.optim.Adam(model_2d.parameters(),lr = lr, weight_decay=1e-8 )
    
    lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2d, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )
    
    if resume:
        # model_direct.load_state_dict(torch.load("./logs/models/"+run_name)["model"])
        model_2d.load_state_dict(torch.load("./logs/models/"+run_name)["model"])
        batch_size = torch.load("./logs/models/"+run_name)["batch_size"]
        last_epoch = torch.load("./logs/models/"+run_name)["epoch"]
        

    training_set = H36_dataset(num_cams=num_cameras, subjectp=subjects[0:5], is_train = True, action="") 
    test_set     = H36_dataset(num_cams=num_cameras, subjectp=subjects[5:7] , is_train = False, action="")
    
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
    
    epoch_losses, epoch_metric = list(), list()
    epoch_val_loss, epoch_val_metric  = list(), list()
    
    for epoch in tqdm(range(n_epochs),desc="Training"):
        train_loss = 0.0
        train_metric = torch.zeros(num_of_joints).to(device)

        # model_direct.train()  
        model_2d.train()
            
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=True, position=0):
            
            # optimizer.zero_grad()
            optimizer_2d.zero_grad()

            x, y, frame  = batch
            
            x,y=x.float(),y.float()
            x, y = x.to(device), y.to(device) 
            frame = frame.float()
            frame =frame.to(device)

            
            # y_hat = model_direct(frame) 
               
            x_hat = model_2d(frame)  
            x_hat = x_hat.reshape(-1,num_of_joints,2) #output_dimension
            
            loss = loss_function(x_hat, x) 
            
            loss.backward()
            # optimizer.step()
            optimizer_2d.step()
            
            # if standardize_3d : 
            #     if Normalize:
            #         y = torch.mul(y , max_train_3d-min_train_3d ) + min_train_3d 
            #         y_hat = torch.mul(y_hat,  max_train_3d-min_train_3d ) + min_train_3d 
            #     else:
            #         temp_std = torch.from_numpy(std_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)
            #         temp_mean = torch.from_numpy(mean_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)
            #         y = torch.mul(y , temp_std ) + temp_mean #DeStandardize
            #         y_hat = torch.mul(y_hat, temp_std ) + temp_mean
            
            train_loss += loss.cpu().item() / len(train_loader)
            train_metric += loss_MPJPE(x_hat, x)/ len(training_set)
            
            
        train_metric = torch.mean(train_metric) #Please be carefull that here we will have zero for the first joint error so maybe it shoudl be the mean over 1:
        if num_of_joints==17 and zero_centre:
                train_metric *= (17/16)*1000
                
            
        lr_schdlr.step(loss)
        
        epoch_losses.append(train_loss)
        epoch_metric.append(train_metric.cpu().item() )
        
        
        #________________validation_______________________  
        with torch.no_grad():
            # model_direct.eval()
            model_2d.eval()
            val_loss = 0.0
            val_metric = torch.zeros(num_of_joints).to(device)
            
            for x_v, y_v, frame_v  in test_loader:
                
                x_v,y_v=x_v.float(),y_v.float()
                x_v, y_v = x_v.to(device), y_v.to(device)
                
                frame_v = frame_v.float()
                frame_v =frame_v.to(device)
                
                # y_hat_v = model_direct(frame_v)
                x_hat_v = model_2d(frame_v)
                    
                x_hat_v = x_hat_v.reshape(-1,num_of_joints,2)
                
                loss_v = loss_function(x_hat_v, x_v) 
                
            
                # if standardize_3d :
                #     if Normalize:
                #         y_v = torch.mul(y_v , max_train_3d-min_train_3d ) + min_train_3d 
                #         y_hat_v = torch.mul(y_hat_v,  max_train_3d-min_train_3d ) + min_train_3d 
                #     else:
                #         y_v = torch.mul(y_v , temp_std ) + temp_mean #DeStandardize
                #         y_hat_v = torch.mul(y_hat_v, temp_std ) + temp_mean   
                    
                    
                metric_v = loss_MPJPE(x_hat_v, x_v) 
            
                val_loss += loss_v.cpu().item() / len(test_loader)
                val_metric += (metric_v / len(test_set))
            
        val_metric = torch.mean(val_metric)
        if num_of_joints==17 and zero_centre:
            val_metric *= (17/16)*1000
        
        epoch_val_loss.append(val_loss)
        epoch_val_metric.append(val_metric.cpu().item() )
        
        if WandB:             
            wandb.log({"loss(train)": train_loss, "loss(val.)": val_loss,"MPJPE(train)":train_metric.cpu().item() , " MPJPE(val.)":val_metric.cpu().item()})   
           
        print(f"epoch {epoch+1}/{n_epochs} loss(train): {train_loss:.4f} , MPJPE(train):{train_metric.cpu().item()}, loss(val.): {val_loss}, MPJPE(val.){val_metric.cpu().item()}") 
        
    # y = y.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    # y_hat = y_hat.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    # visualize_3d(y[0],y_hat[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_a.png")
    # visualize_3d(y[-1],y_hat[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_b.png")
    x = x.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    x_hat = x_hat.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    frame = frame.cpu().detach().numpy()
    visualize_2d(x[0],x_hat[0],frame[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_train_a.png")
    visualize_2d(x[-1],x_hat[-1],frame[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_train_b.png")     
    
    plot_losses(epoch_losses,epoch_val_loss,epoch_metric,epoch_val_metric,"./logs/visualizations/"+(resume*"resumed_")+run_name)
    
    # y_v = y_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    # y_hat_v = y_hat_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    # visualize_3d(y_v[0],y_hat_v[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_test_a.png")
    # visualize_3d(y_v[-1],y_hat_v[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_test_b.png")
    x_v = x_v.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    x_hat_v = x_hat_v.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    frame_v = frame_v.cpu().detach().numpy()
    visualize_2d(x_v[0],x_hat_v[0],frame_v[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_test_a.png")
    visualize_2d(x_v[-1],x_hat_v[-1],frame_v[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_test_b.png")            
    
    torch.save({'epoch' : epoch, 'batch_size':batch_size, 'model' : model_2d.state_dict(), 'optimizer': optimizer_2d.state_dict() , 'scheduler': lr_schdlr.state_dict() },"./logs/models/"+(resume*"resumed_")+run_name)
    
    return model_2d


def infer(run_name):
    
    model_direct= Model_3D().to(device)
    model_direct.load_state_dict(torch.load("./logs/models/"+run_name)["model"])
    
    test_set     = H36_dataset(num_cams=num_cameras, subjectp=subjects[5:7] , is_train = False, action="Walking 1")
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=1)
 
    mean_train_3d, std_train_3d = load_statisctics("mean_train_3d"), load_statisctics("std_train_3d")
    max_train_3d, min_train_3d = load_statisctics("max_train_3d"), load_statisctics("min_train_3d")
    max_train_3d, min_train_3d = torch.from_numpy(max_train_3d).to(device), torch.from_numpy(min_train_3d).to(device)
    if zero_centre and num_of_joints==17 : 
        max_train_3d[:1,:] *= 0 
        min_train_3d[:1,:] *= 0
    
    loss_function = torch.nn.MSELoss(reduction = "mean")    
        
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
            
            loss_v = loss_function(x_hat_v, x_v) 
            
            # if standardize_3d :
            #     if Normalize:
            #         y_v = torch.mul(y_v , max_train_3d-min_train_3d ) + min_train_3d 
            #         y_hat_v = torch.mul(y_hat_v,  max_train_3d-min_train_3d ) + min_train_3d 
                # else:
                #     y_v = torch.mul(y_v , temp_std ) + temp_mean #DeStandardize
                #     y_hat_v = torch.mul(y_hat_v, temp_std ) + temp_mean   
                                    
            metric_v = loss_MPJPE(x_hat_v, x_v) 
        
            val_loss += loss_v.cpu().item() / len(test_loader)
            val_metric += (metric_v / len(test_set))
            
        val_metric = torch.mean(val_metric)
        if num_of_joints==17 and zero_centre:
            val_metric *= (17/16)*1000


        print(f"loss(test): {val_loss}, MPJPE(test){val_metric.cpu().item()}") 

        # y_v = y_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
        # y_hat_v = y_hat_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
        # visualize_3d(y_v[0],y_hat_v[0],   "./logs/visualizations/"+str(run_name)+"/"+"infer_"+"resumed_"+"3d_test_a.png")
        # visualize_3d(y_v[-1],y_hat_v[-1], "./logs/visualizations/"+str(run_name)+"/"+"infer_"+"resumed_"+"3d_test_b.png")     

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",device)
    batch_size = 64
    n_epochs= 5
    lr = 0.005 #0.001
    run_name = "test"
    CtlCSave = False
    Resume = False
    Train = True
    
    if Train :
        print("___"+run_name+"___")
        if not os.path.exists("./logs/visualizations/"+run_name):
            os.mkdir(os.path.join("./logs/visualizations/", run_name))
            
        if WandB:
            wandb.init(project="Direct_2D_Pose",name=run_name, config={"learning_rate": lr, "architecture": "CNN","dataset": "H3.6","epochs": n_epochs,})
        
        try:
            model = train(batch_size,n_epochs,lr,device,run_name,resume=Resume)
            torch.save(model.state_dict(),"./logs/models/second_"+run_name)
        except KeyboardInterrupt:
                if CtlCSave: torch.save(model.state_dict(),"./logs/models/interrupt_"+run_name)
        
        if WandB:
            wandb.finish()
        print("___"+run_name+" DONE___")
    
    else: 
        infer(run_name)