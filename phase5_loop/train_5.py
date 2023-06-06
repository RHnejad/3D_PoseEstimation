import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb
WandB = False

from Model_2d import Model_2D

import sys
sys.path.append("../phase3_direct/my_HybrIK/")
from utils import visualize_3d,visualize_2d, plot_losses
from Model import Model_3D
from H36_dataset import *

sys.path.append("../phase1_lifting/")
from baselineModel import LinearModel

def loss_MPJPE(prediction, target):
    B,J,d =  target.shape
    position_error = torch.norm(prediction - target, dim=-1) #len(target.shape)-1
    metric = torch.sum(position_error, dim=0)
    return metric

def load_statisctics(file_name):
    with open("./logs/run_time_utils/"+file_name+".npy","rb") as f:
        array =np.load(f)
    return array 


class TriangleLoss(torch.nn.Module):
    def __init__(self):
        super(TriangleLoss, self).__init__()
        self.loss_2d = []
        self.loss_3d = []
        self.loss_lift = []
        self.loss_domain_gap = []
        
        self.loss_function = torch.nn.L1Loss()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, predicted_2d, predicted_3d, lift_2d_gt, lift_2d_pred , gt_2d, gt_3d):
        
        loss_2d_ = self.loss_function(predicted_2d, gt_2d) 
        loss_3d_ = self.loss_function(predicted_3d, gt_3d)
         
        domain_gap_loss = self.loss_function(lift_2d_pred, lift_2d_gt)
        loss_lift = self.loss_function(lift_2d_gt, gt_3d)
        
        self.loss_2d.append(loss_2d_.cpu().item())
        self.loss_3d.append(loss_3d_.cpu().item())
        self.loss_lift.append(loss_lift.cpu().item())
        self.loss_domain_gap.append(domain_gap_loss)
        # print(loss_2d_ , loss_3d_ , loss_lift_)
        # breakpoint()
        return (loss_2d_ + loss_3d_ + loss_lift + domain_gap_loss)
    
    def report_losses(self):
        print(self.loss_2d[-1] , self.loss_3d[-1] , self.loss_lift[-1], self.loss_domain_gap[-1])
        return self.loss_2d, self.loss_3d, self.loss_lift
    

def train(batch_size,n_epochs,lr,device,run_name,resume=False, Triangle=True):
    
    #Creating Models
    model_2d= Model_2D().to(device)
    model_3d= Model_3D().to(device)
    model_lift = LinearModel(i_dim=num_of_joints*input_dimension, o_dim=num_of_joints*output_dimension,p_dropout=0.5, linear_size=1024).to(device)
    
    if Triangle:
        loss_function = TriangleLoss()
    else:
        loss_function = torch.nn.L1Loss()
        loss_function = torch.nn.MSELoss(reduction = "mean")
        
    # optimizer = torch.optim.Adam(model_direct.parameters(),lr = lr)
    
    optimizer_2d = torch.optim.AdamW(model_2d.parameters(),lr = lr)#, weight_decay=1e-8 
    optimizer_3d = torch.optim.AdamW(model_3d.parameters(),lr = lr)
    optimizer_lift = torch.optim.AdamW(model_lift.parameters(),lr = lr )
    
    # lr_schdlr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_3d, factor=0.7, patience=3, cooldown=2, min_lr=5e-6, verbose=True )
    
    if resume:
        # model_direct.load_state_dict(torch.load("./logs/models/"+run_name)["model"])
        model_2d.load_state_dict(torch.load("./logs/models/"+run_name)["model"])
        batch_size = torch.load("./logs/models/"+run_name)["batch_size"]
        last_epoch = torch.load("./logs/models/"+run_name)["epoch"]
        

    training_set = H36_dataset(num_cams=num_cameras, subjectp=subjects[0:5], is_train = True, action="") #new
    test_set     = H36_dataset(num_cams=num_cameras, subjectp=subjects[5:7] , is_train = False, action="")
    
    train_loader = DataLoader( training_set, shuffle=True, batch_size=batch_size, num_workers= 1)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=1)
   
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
            
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=True, position=0):

            optimizer_3d.zero_grad()
            optimizer_2d.zero_grad()
            optimizer_lift.zero_grad()

            y1, y2, frame, _  = batch
            current_batch_size = y1.shape[0]
            
            y1,y2=y1.float(),y2.float()
            y1, y2 = y1.to(device), y2.to(device) 
            frame = frame.float()
            frame =frame.to(device)
                           
            y1_hat = model_2d(frame).reshape(current_batch_size,num_of_joints,2)   
            y2_hat = model_3d(frame).reshape(current_batch_size,num_of_joints,3)
            lift_2d_pred = model_lift(y1_hat).reshape(current_batch_size,num_of_joints,3)
            lift_2d_gt = model_lift(y1).reshape(current_batch_size,num_of_joints,3)
            
    
            if Triangle: 
                #predicted_2d, predicted_3d, lift_2d_gt, lift_2d_pred , gt_2d, gt_3d
                loss = loss_function(predicted_2d = y1_hat, predicted_3d = y2_hat,
                                     lift_2d_gt = lift_2d_gt, lift_2d_pred=lift_2d_pred,
                                     gt_2d = y1, gt_3d = y2)
                loss.backward()  
            else :
                pass
                loss_2d = loss_function(y1_hat, y1) 
                loss_3d = loss_function(y2_hat, y2) 
                # loss_lift = loss_function(y2_hat_lift, y2_hat) 
    
                loss_2d.backward()
                loss_3d.backward()
                # loss_lift.backward()


            optimizer_2d.step()
            optimizer_3d.step()
            optimizer_lift.step()
            
            # if standardize_3d : 
            #     if Normalize:
            #         y = torch.mul(y , max_train_3d-min_train_3d ) + min_train_3d 
            #         y_hat = torch.mul(y_hat,  max_train_3d-min_train_3d ) + min_train_3d 
            #     else:
            #         temp_std = torch.from_numpy(std_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)
            #         temp_mean = torch.from_numpy(mean_k).to(device).expand(y.shape[0],num_of_joints,output_dimension)
            #         y = torch.mul(y , temp_std ) + temp_mean #DeStandardize
            #         y_hat = torch.mul(y_hat, temp_std ) + temp_mean
            
            if Triangle:
                train_loss += loss.cpu().item() / len(train_loader)
            else:
                train_loss += loss_2d.cpu().item() / len(train_loader)
                
            train_metric_3d += loss_MPJPE(y2_hat, y2)/ len(training_set)
            
            
        train_metric_3d = torch.mean(train_metric_3d) #Please be carefull that here we will have zero for the first joint error so maybe it shoudl be the mean over 1:
        if num_of_joints==17 and zero_centre:
                train_metric_3d *= (17/16)*1000
                
            
        # lr_schdlr.step(loss) #fix this for both 2d and 3d
        
        epoch_losses.append(train_loss)
        epoch_metric.append(train_metric_3d.cpu().item() )
        
        
        #________________validation_______________________  
        with torch.no_grad():

            model_2d.eval()
            model_3d.eval()
            model_lift.eval()
            
            val_loss = 0.0
            val_metric_3d = torch.zeros(num_of_joints).to(device)
            
            for y1_v, y2_v, frame_v, _  in test_loader:
                
                current_batch_size = y1_v.shape[0]
                
                y1_v,y2_v=y1_v.float(),y2_v.float()
                y1_v, y2_v = y1_v.to(device), y2_v.to(device)
                
                frame_v = frame_v.float()
                frame_v =frame_v.to(device)
                
                y1_hat_v = model_2d(frame_v).reshape(current_batch_size,num_of_joints,2)

                y2_hat_v = model_3d(frame_v).reshape(current_batch_size,num_of_joints,3)

                lift_2d_pred_v = model_lift(y1_hat_v).reshape(current_batch_size,num_of_joints,3)
                lift_2d_gt_v = model_lift(y1_v).reshape(current_batch_size,num_of_joints,3)

                #predicted_2d, predicted_3d, lift_2d_gt, lift_2d_pred , gt_2d, gt_3d
                if Triangle:
                    loss_v = loss_function(predicted_2d = y1_hat_v, predicted_3d = y2_hat_v,
                                    lift_2d_gt = lift_2d_gt_v , lift_2d_pred = lift_2d_pred_v,
                                    gt_2d = y1_v, gt_3d = y2_v)
                else :
                    loss_2d_v = loss_function(y1_hat_v, y1_v) 
                    loss_3d_v = loss_function(y2_hat_v, y2_v) 
                    
                
            
                # if standardize_3d :
                #     if Normalize:
                #         y2_v = torch.mul(y2_v , max_train_3d-min_train_3d ) + min_train_3d 
                #         y2_hat_v = torch.mul(y2_hat_v,  max_train_3d-min_train_3d ) + min_train_3d 
                #     else:
                #         y2_v = torch.mul(y2_v , temp_std ) + temp_mean #DeStandardize
                #         y2_hat_v = torch.mul(y2_hat_v, temp_std ) + temp_mean   
                    
                    
                metric_v_3d = loss_MPJPE(y2_hat_v, y2_v) 
            
                if Triangle:
                    val_loss += loss_v.cpu().item() / len(test_loader)
                else:
                    val_loss += loss_2d_v.cpu().item() / len(test_loader)
                val_metric_3d += (metric_v_3d / len(test_set))
            
        val_metric_3d = torch.mean(val_metric_3d)
        if num_of_joints==17 and zero_centre:
            val_metric_3d *= (17/16)*1000
        
        epoch_val_loss.append(val_loss)
        epoch_val_metric.append(val_metric_3d.cpu().item() )
        
        if WandB:             
            wandb.log({"loss(train)": train_loss, "loss(val.)": val_loss,"MPJPE(train)":train_metric_3d.cpu().item() , " MPJPE(val.)":val_metric_3d.cpu().item()})   
        
        if Triangle:
            print("___losses___")   
            loss_function.report_losses()
 
            
        print(f"epoch {epoch+1}/{n_epochs} loss(train): {train_loss:.4f} , MPJPE(train):{train_metric_3d.cpu().item()}, loss(val.): {val_loss}, MPJPE(val.){val_metric_3d.cpu().item()}") 
        
    
    #___visualize__train___
        
    y2 = y2.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y2_hat = y2_hat.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y2[0],y2_hat[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_a.png")
    visualize_3d(y2[-1],y2_hat[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_b.png")
        
    lift_2d_pred = lift_2d_pred.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y2[0],lift_2d_pred[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_lift_train_a.png")
    visualize_3d(y2[-1],lift_2d_pred[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_lift_train_b.png")    
    
    y1 = y1.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    y1_hat = y1_hat.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    frame = frame.cpu().detach().numpy()
    visualize_2d(y1[0],y1_hat[0],frame[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_train_a.png")
    visualize_2d(y1[-1],y1_hat[-1],frame[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_train_b.png")     
    
    plot_losses(epoch_losses,epoch_val_loss,epoch_metric,epoch_val_metric,"./logs/visualizations/"+(resume*"resumed_")+run_name)
    
    #___visualize__validation___
    
    y2_v = y2_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y2_hat_v = y2_hat_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y2_v[0],y2_hat_v[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_test_a.png")
    visualize_3d(y2_v[-1],y2_hat_v[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_test_b.png")
    
    lift_2d_pred_v = lift_2d_pred_v.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y2_v[0],lift_2d_pred_v[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_lift_test_a.png")
    visualize_3d(y2_v[-1],lift_2d_pred_v[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_lift_test_b.png")
    
    y1_v = y1_v.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    y1_hat_v = y1_hat_v.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    frame_v = frame_v.cpu().detach().numpy()
    visualize_2d(y1_v[0],y1_hat_v[0],frame_v[0],   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_test_a.png")
    visualize_2d(y1_v[-1],y1_hat_v[-1],frame_v[-1], "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_test_b.png")            
    
    #, 'scheduler': lr_schdlr.state_dict()
    torch.save({'epoch' : epoch, 'batch_size':batch_size, 'model' : model_2d.state_dict(), 'optimizer': optimizer_2d.state_dict()  },"./logs/models/"+(resume*"resumed_")+run_name)
    
    return model_2d, model_3d, model_lift


def infer(run_name): 
    pass
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",device)
    batch_size = 16
    n_epochs= 20
    lr = 0.001#0.001
    run_name = "test_resnet101_TR"
    CtlCSave = False
    Resume = False
    Train = True
    Triangle = True
    
    if Train :
        print("___"+run_name+"___")
        if not os.path.exists("./logs/visualizations/"+run_name):
            os.mkdir(os.path.join("./logs/visualizations/", run_name))
            
        if WandB:
            wandb.init(project="Direct_2D_Pose",name=run_name, config={"learning_rate": lr, "architecture": "CNN","dataset": "H3.6","epochs": n_epochs,})
        
        # try:
        model = train(batch_size,n_epochs,lr,device,run_name,resume=Resume, Triangle=Triangle)
        # torch.save(model.state_dict(),"./logs/models/second_"+run_name)
        # except KeyboardInterrupt:
        #         if CtlCSave: torch.save(model.state_dict(),"./logs/models/interrupt_"+run_name)
        
        if WandB:
            wandb.finish()
        print("___"+run_name+" DONE___")
    
    else: 
        infer(run_name)