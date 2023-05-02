from Model import Model_3D
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from H36_dataset import *
from tqdm import tqdm

def loss_MPJPE(prediction, target):
    B,J,d =  target.shape
    position_error = torch.norm(prediction - target, dim=-1) #len(target.shape)-1
    metric = torch.sum(position_error, dim=0)
    return metric


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",device)
    batch_size = 64
    n_epochs=20
    lr = 0.001
    
    training_set = H36_dataset(num_cams=num_cameras, subjectp=subjects[0:1], is_train = True) 
    test_set     = H36_dataset(num_cams=num_cameras, subjectp=subjects[0:1] , is_train = False)
    
    train_loader = DataLoader( training_set, shuffle=True, batch_size=batch_size, num_workers= 1)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=1)

    #loading mean and std
    with open("mean_train_3d.npy","rb") as f:
        mean_train_3d =np.load(f)  
    with open("std_train_3d.npy","rb") as f:
        std_train_3d = np.load(f)  
    with open("max_train_3d.npy","rb") as f:
        max_train_3d = torch.from_numpy(np.load(f)).to(device)  
    with open("min_train_3d.npy","rb") as f:
        min_train_3d = torch.from_numpy(np.load(f)).to(device)   

    mean_k = mean_train_3d [list(range(17-num_of_joints,17)),:]
    std_k = std_train_3d [list(range(17-num_of_joints,17)),:]    
    
    model_direct= Model_3D().to(device)

    loss_function = torch.nn.MSELoss(reduction = "mean")
    optimizer = torch.optim.Adam(model_direct.parameters(),lr = lr)
    
    lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )
    
    epoch_losses = list()
    epoch_metric = list()
    epoch_eval_loss = list()
    epoch_eval_metric = list()
    

    for epoch in tqdm(range(n_epochs),desc="Training"):
        train_loss = 0.0
        train_metric = torch.zeros(num_of_joints).to(device)

        model_direct.train()  
            
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=False):
            
            optimizer.zero_grad()

            x, y, frame  = batch
            
            x, y = x.to(device), y.to(device)
            x,y=x.float(),y.float()
            
            frame =frame.to(device)
            frame = frame.float()
            y_hat = model_direct(frame)
                  
            y_hat = y_hat.reshape(-1,num_of_joints,output_dimension)
            
            loss = loss_function(y_hat, y) 
            loss.backward()
            optimizer.step()
            
            temp_std = torch.from_numpy(std_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)
            temp_mean = torch.from_numpy(mean_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)

            if standardize_3d :
                if Normalize:
                    y = torch.mul(y , max_train_3d-min_train_3d ) + min_train_3d 
                    y_hat = torch.mul(y_hat,  max_train_3d-min_train_3d ) + min_train_3d 
                else:
                    y = torch.mul(y , temp_std ) + temp_mean #DeStandardize
                    y_hat = torch.mul(y_hat, temp_std ) + temp_mean
            
            train_loss += loss.cpu().item() / len(train_loader)
            train_metric += loss_MPJPE(y_hat, y)/ len(training_set)
            
        train_metric = torch.mean(train_metric)
            
        lr_schdlr.step(loss)
                
        print(f"epoch {epoch+1}/{n_epochs} loss(train): {train_loss:.4f} , MPJPE(train):{train_metric.cpu().item()}") 
