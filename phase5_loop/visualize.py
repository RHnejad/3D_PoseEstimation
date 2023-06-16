
import torch

sys.path.append("../phase3_direct/my_HybrIK/")
from utils import visualize_3d,visualize_2d, plot_losses

num_of_joints= 17
output_dimension = 3

def visualize(y1,y2,y1_hat,h2_hat,frame,run_name, resume):
    
    y2 = y2.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    y2_hat = y2_hat.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
    visualize_3d(y2[0].copy(),y2_hat[0].copy(),   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_a.png")
    visualize_3d(y2[-1].copy(),y2_hat[-1].copy(), "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_train_b.png")
        
    try:    
        lift_2d_pred = lift_2d_pred.cpu().detach().numpy().reshape(-1, num_of_joints,output_dimension)
        visualize_3d(y2[0].copy(),lift_2d_pred[0].copy(),   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_lift_train_a.png")
        visualize_3d(y2[-1].copy(),lift_2d_pred[-1].copy(), "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_lift_train_b.png") 
        
    except:
        print("NO 2D to 3D LIFTING RESULTS TO PLOT")   
    
    y1 = y1.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    y1_hat = y1_hat.cpu().detach().numpy().reshape(-1, num_of_joints,2)
    frame = torch.permute(frame, (0,2,3,1))
    frame = frame.cpu().detach().numpy()
    visualize_2d(y1[0].copy(),y1_hat[0].copy(),frame[0].copy(),   "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_train_a.png")
    visualize_2d(y1[-1].copy(),y1_hat[-1].copy(),frame[-1].copy(), "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"2d_train_b.png")     
    
    try:
        proj_3d_pred = proj_3d_pred.cpu().detach().numpy().reshape(-1, num_of_joints,2)
        visualize_2d(y1[0].copy(),proj_3d_pred[0].copy(),  frame[0].copy(), "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_proj_train_a.png")
        visualize_2d(y1[-1].copy(),proj_3d_pred[-1].copy(),frame[-1].copy(), "./logs/visualizations/"+str(run_name)+"/"+resume*"resumed_"+"3d_proj_train_b.png") 
        
    except:
        print("NO 3D to 2D PROJECTION RESULTS TO PLOT") 
    
    
    