import torch 
import torch.nn as nn
from torch.nn import functional as F
import torch.cuda.comm

from Resnet import ResNet
from utils import visualize_3d,plot_heat_map
#https://github.dev/Jeff-sjtu/HybrIK/tree/b8cfeeb7df2d7c8024865751cc17664a8272557b/hybrik/models


class Model_3D(nn.Module):
    def __init__(self):
        super(Model_3D, self).__init__()
        
        #_________
        self.deconv_dim = [256,256,256] # kwargs['NUM_DECONV_FILTERS']
        self.num_joints = 17 #kwargs['NUM_JOINTS']
        self.depth_dim = 64 #kwargs['EXTRA']['DEPTH_DIM']
        self.norm_type = 'softmax'
        self.height_dim = 64
        self.width_dim = 64
        
        self._norm_layer = nn.BatchNorm2d
        
        #__________preact___________
        self.preact = ResNet("resnet50") 
        
        # Imagenet pretrain model
        import torchvision.models as tm   
        x = tm.resnet50(pretrained=True)
        self.feature_channel = 2048
        
        model_state = self.preact.state_dict()
        state = {k: v for k, v in x.state_dict().items()
                 if k in self.preact.state_dict() and v.size() == self.preact.state_dict()[k].size()}
        model_state.update(state)
        self.preact.load_state_dict(model_state)
           
        #________deconv_layers______
        # self.deconv_layers = self._make_deconv_layer_with_dropout()
        self.deconv_layers = self._make_deconv_layer()
        
        #________final_layer________
        self.final_layer = nn.Conv2d(
            self.deconv_dim[2], self.num_joints * self.depth_dim, kernel_size=1, stride=1, padding=0)
    
    
    def _make_deconv_layer(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)
    
    def _make_deconv_layer_with_dropout(self):
        deconv_layers = []
        deconv1 = nn.ConvTranspose2d(
            self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn1 = self._norm_layer(self.deconv_dim[0])
        deconv2 = nn.ConvTranspose2d(
            self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn2 = self._norm_layer(self.deconv_dim[1])
        deconv3 = nn.ConvTranspose2d(
            self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
        bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(nn.Dropout2d(p=0.5))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(nn.Dropout2d(p=0.5))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(nn.Dropout2d(p=0.5))

        return nn.Sequential(*deconv_layers)
    

    def norm_heatmap(self, norm_type, heatmap ):
        # Input tensor shape: [N,C,...]
        shape = heatmap.shape
        if norm_type == 'softmax':
            heatmap = heatmap.reshape(*shape[:2], -1)
            # global soft max
            heatmap = F.softmax(heatmap, 2)
            return heatmap.reshape(*shape)
        else:
            raise NotImplementedError
        
    def uvw_to_xyz(self, kp):
        return torch.tensor([kp[2], -kp[0], -kp[1]])
            
    def forward(self, x):
    
        batch_size = x.shape[0]
    
        x0 = self.preact(x)
        out = self.deconv_layers(x0)
        out = self.final_layer(out)
        
        returned_heatmap = out.clone().detach()
         
        out = out.reshape((out.shape[0], self.num_joints, -1))
        out = self.norm_heatmap(self.norm_type, out)
        assert out.dim() == 3, out.shape
        
        retuned_heatmap = out.reshape((-1,17,64,64,64))
        
        # maxvals = torch.ones((*out.shape[:2], 1), dtype=torch.float, device=out.device) #check

        heatmaps = out / out.sum(dim=2, keepdim=True) #out.sum is 1 if softmax so not actually needed
        
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, self.depth_dim, self.height_dim, self.width_dim))
        
        
        hm_x = heatmaps.sum((2, 3))
        hm_y = heatmaps.sum((2, 4))
        hm_z = heatmaps.sum((3, 4))
        
 
        if torch.cuda.is_available():
            hm_x = hm_x * torch.cuda.comm.broadcast(torch.arange(hm_x.shape[-1]).type(
                torch.cuda.FloatTensor), devices=[hm_x.device.index])[0]
            hm_y = hm_y * torch.cuda.comm.broadcast(torch.arange(hm_y.shape[-1]).type(
                torch.cuda.FloatTensor), devices=[hm_y.device.index])[0]
            hm_z = hm_z * torch.cuda.comm.broadcast(torch.arange(hm_z.shape[-1]).type(
                torch.cuda.FloatTensor), devices=[hm_z.device.index])[0]
        else :
            hm_x = hm_x * torch.arange(hm_x.shape[-1]).type(torch.FloatTensor)
            hm_y = hm_y * torch.arange(hm_y.shape[-1]).type(torch.FloatTensor)
            hm_z = hm_z * torch.arange(hm_z.shape[-1]).type(torch.FloatTensor)
              
            
        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)

        coord_x = (coord_x / float(self.width_dim) - 0.5)*2
        coord_y = (coord_y / float(self.height_dim) - 0.5)*2
        coord_z = (coord_z / float(self.depth_dim) - 0.5)*2
        
        #  -0.5 ~ 0.5 0-1
        pred_uvd_jts_29 = torch.cat((coord_x, coord_y, coord_z), dim=2)
        
        # pred_uvd_jts_29 = torch.cat((coord_x, -coord_z, -coord_y), dim=2) #HERE

        pred_uvd_jts_29_flat = pred_uvd_jts_29.reshape((batch_size, self.num_joints * 3)) 
                
        return pred_uvd_jts_29_flat, retuned_heatmap
    
    
if __name__ == "__main__":
    from H36_dataset import H36_dataset
    
    from torch.utils.data import DataLoader
    training_set = H36_dataset(subjectp=["S1"], is_train = True) 
    train_loader = DataLoader( training_set, shuffle=True, batch_size=1, num_workers= 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:",device)
    x,y,f,_ = next(iter(train_loader))
    f = f.float().to(device)
    f= torch.permute(f, (0,3,1,2))
    model = Model_3D().to(device)
    
    z = model(f)
