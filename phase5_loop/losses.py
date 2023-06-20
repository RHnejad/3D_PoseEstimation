import torch

def loss_MPJPE(prediction, target):
    B,J,d =  target.shape
    position_error = torch.norm(prediction - target, dim=-1) #len(target.shape)-1
    metric = torch.sum(position_error, dim=0)
    return metric


class TriangleLoss(torch.nn.Module):
    def __init__(self, Project = False):
        super(TriangleLoss, self).__init__()
        
        self.Project = Project
        
        self.loss_2d = []
        self.loss_3d = []
        self.loss_lift = []
        self.loss_proj = []
               
        self.loss_function = torch.nn.L1Loss()
        # self.loss_function = torch.nn.MSELoss()

    def forward(self, predicted_2d, predicted_3d, lift_2d_gt, lift_2d_pred , gt_2d, gt_3d , proj_3d_pred=None, proj_3d_gt=None):
        
        loss_2d_ = self.loss_function(predicted_2d, gt_2d) 
        loss_3d_ = self.loss_function(predicted_3d, gt_3d)
         
        # domain_gap_loss = self.loss_function(lift_2d_pred, lift_2d_gt)
        loss_lift = self.loss_function(lift_2d_pred, predicted_3d)
        loss_proj = 0
        if self.Project:
            try:
                proj_3d_pred[1:] -= proj_3d_pred[0]
                # proj_3d_gt[1:] -= proj_3d_gt[0]
                predicted_2d_ = predicted_2d.clone()
                predicted_2d_[1:] -= predicted_2d_[0]
                
                # loss_gap_proj = self.loss_function(proj_3d_pred, proj_3d_gt)  
                loss_proj = self.loss_function(proj_3d_pred, predicted_2d_)
            except:
                breakpoint()
             
            returned_loss = loss_2d_ + loss_3d_ + loss_lift + loss_proj 
        else:
            returned_loss = loss_2d_ + loss_3d_ + loss_lift
            
        self.loss_2d.append(loss_2d_.cpu().item())
        self.loss_3d.append(loss_3d_.cpu().item())
        self.loss_lift.append(loss_lift.cpu().item())
        if self.Project : self.loss_proj.append(loss_proj.cpu().item())
        
        return returned_loss, loss_2d_, loss_3d_, loss_lift, loss_proj 
    
    def report_losses(self):
        print(sum(self.loss_2d)/len(self.loss_2d) , sum(self.loss_3d)/len(self.loss_3d) ,
              sum(self.loss_lift)/len(self.loss_lift) )
        
        self.loss_2d = []
        self.loss_3d = []
        self.loss_lift = []
        self.loss_domain_gap = []


class TriangleLoss_sep(torch.nn.Module):
    def __init__(self, Project = False):
        super(TriangleLoss_sep, self).__init__()
        
        self.Project = Project
        
        self.loss_2d = []
        self.loss_3d = []
        self.loss_lift = []
        self.loss_domain_gap = []
        self.loss_proj = []
        self.loss_gap_proj = [] 
               
        self.loss_function = torch.nn.L1Loss()
        # self.loss_function = torch.nn.MSELoss()

    def forward(self, predicted_2d, predicted_3d, lift_2d_gt, lift_2d_pred , gt_2d, gt_3d , proj_3d_pred=None, proj_3d_gt=None):
        
        loss_2d_ = self.loss_function(predicted_2d, gt_2d) 
        loss_3d_ = self.loss_function(predicted_3d, gt_3d)
         
        domain_gap_loss = self.loss_function(lift_2d_pred, lift_2d_gt)
        loss_lift = self.loss_function(lift_2d_gt, gt_3d)
        
        if self.Project:
            try:
                proj_3d_pred[1:] -= proj_3d_pred[0]
                proj_3d_gt[1:] -= proj_3d_gt[0]
                gt_2d_c = gt_2d.clone()
                gt_2d_c[1:] -= gt_2d_c[0]
                
                loss_gap_proj = self.loss_function(proj_3d_pred, proj_3d_gt)  
                loss_proj = self.loss_function(proj_3d_gt, gt_2d_c)
            except:
                breakpoint()
             
            returned_loss = loss_2d_ + loss_3d_ + loss_lift + domain_gap_loss + loss_proj + loss_gap_proj
        else:
            returned_loss = loss_2d_ + loss_3d_ + loss_lift + domain_gap_loss
            
        self.loss_2d.append(loss_2d_.cpu().item())
        self.loss_3d.append(loss_3d_.cpu().item())
        self.loss_lift.append(loss_lift.cpu().item())
        self.loss_domain_gap.append(domain_gap_loss.cpu().item())
        
        return returned_loss
    
    def report_losses(self):
        print(sum(self.loss_2d)/len(self.loss_2d) , sum(self.loss_3d)/len(self.loss_3d) ,
              sum(self.loss_lift)/len(self.loss_lift), sum(self.loss_domain_gap)/len(self.loss_domain_gap))
        
        self.loss_2d = []
        self.loss_3d = []
        self.loss_lift = []
        self.loss_domain_gap = []   