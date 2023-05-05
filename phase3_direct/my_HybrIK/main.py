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

class _3D_HPE_():
    def __init__(self,batch_size,n_epochs,lr, device, run_name= "" ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = device
        self.run_name = run_name
        
        self.training_set = H36_dataset(num_cams=num_cameras, subjectp=subjects[0:5], is_train = True) 
        self.test_set     = H36_dataset(num_cams=num_cameras, subjectp=subjects[5:7] , is_train = False)
        
        self.train_loader = DataLoader( self.training_set, shuffle=True, batch_size=batch_size, num_workers= 1)
        self.test_loader = DataLoader(self.test_set, shuffle=True, batch_size=batch_size, num_workers=1)

        #loading mean and std
        self.mean_train_3d =self.load_statistics("mean_train_3d")  
        self.std_train_3d = self.load_statistics("std_train_3d")
            
        self.max_train_3d = torch.from_numpy(self.load_statistics("max_train_3d")).to(device) 
        if zero_centre and num_of_joints==17:
            self.max_train_3d[:1,:]*=0 
            
        self.min_train_3d = torch.from_numpy(np.load(self.load_statistics("min_train_3d"))).to(device)
        if zero_centre and num_of_joints==17:  
            self.min_train_3d[:1,:]*=0 
                
        self.mean_k = self.mean_train_3d [list(range(17-num_of_joints,17)),:]
        self.std_k = self.std_train_3d [list(range(17-num_of_joints,17)),:] 
        
        #creating the model
        self.model_direct= Model_3D().to(device)
        self.loss_function = torch.nn.MSELoss(reduction = "mean")
        self.optimizer = torch.optim.Adam(self.model_direct.parameters(),lr = lr)
        
        self.lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )

        self.epoch_losses = list()
        self.epoch_metric = list()
        self.epoch_val_loss = list()
        self.epoch_val_metric = list() 
        
        self.epoch = None

    
    def load_statistics(self, name):
        with open("./logs/run_time_utils/"+name+".npy","rb") as f:
            array =np.load(f) 
        return array 
    
    def train(self):
        for self.epoch in tqdm(range(self.n_epochs),desc="Training"):
            
            train_loss = 0.0
            train_metric = torch.zeros(num_of_joints).to(self.device)

            self.model_direct.train()  
        
            loss,metric = self.run_batches(self.train_loader, train=True)
            
            self.train_metric = torch.mean(metric)
            #Please be carefull that here we will have zero for the first joint error so maybe it shoudl be the mean over 1:
            if num_of_joints==17 and zero_centre:
                self.train_metric *= (17/16)
                
            epoch_losses.append(train_loss)
            epoch_metric.append(train_metric.cpu().item() )
            loss,metric = self.run_batches(self.test_loader, train=False)
            
    def run_batches(self,data_loader,train=True):
        
        # for batch in tqdm(data_loader, desc=f"Epoch {self.epoch+1} in training", leave=False):
        
        #     self.optimizer.zero_grad()

        #     x, y, frame  = batch
            
        #     x,y=x.float(),y.float()
        #     x, y = x.to(self.device), y.to(self.device)
            
        #     frame = frame.float()
        #     frame =frame.to(self.device)
            
        #     y_hat = self.model_direct(frame)
                
        #     y_hat = y_hat.reshape(-1,num_of_joints,output_dimension)
            
        #     loss = self.loss_function(y_hat, y) 
            
        #     if train:
        #         loss.backward()
        #         self.optimizer.step()
            

        #     y,y_hat =self.de_standardize(y,y_hat)
            
        #     batches_loss += loss.cpu().item() / (len(self.train_loader) if train else len(self.test_loader))
        #     batches_metric += self.cal_MPJPE(y, y_hat)/ (len(self.training_set) if train else len(self.test_set))
            
        # if train:
        #     self.lr_schdlr.step(loss)
            
        return batches_loss, batches_metric
           
    def de_standardize(self, y, y_hat):
        
        if standardize_3d and not Normalize:
            temp_std = torch.from_numpy(self.std_k).to(self.device).expand(y.shape[0],num_of_joints,output_dimension)
            temp_mean = torch.from_numpy(self.mean_k).to(self.device).expand(y.shape[0],num_of_joints,output_dimension)
        
        if standardize_3d :
            if Normalize:
                y = torch.mul(y , self.max_train_3d-self.min_train_3d ) + self.min_train_3d 
                y_hat = torch.mul(y_hat,  self.max_train_3d-self.min_train_3d ) + self.min_train_3d 
            else:
                y = torch.mul(y , temp_std ) + temp_mean #DeStandardize
                y_hat = torch.mul(y_hat, temp_std ) + temp_mean
            
        return y, y_hat

    def test(self):
        pass
    
    
    def cal_MPJPE(self, target, prediction) :
        B,J,d =  target.shape
        position_error = torch.norm(prediction - target, dim=-1) #len(target.shape)-1
        metric = torch.sum(position_error, dim=0)
        return metric
        
        
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    pass