from Model import Model_3D
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from H36_dataset import *

def generate_image():
    img_height = 256 #224
    img_width = 256 #224
    num_channels = 3

    # # Generate a random image with values between 0 and 255
    # random_image = np.random.randint(0, 256, size=(img_height, img_width, num_channels), dtype=np.uint8)

    # # Convert the NumPy array to a PyTorch tensor
    # random_image_tensor = torch.from_numpy(random_image.transpose((2, 0, 1))).float()

    # # Normalize the tensor values to be between 0 and 1
    # random_image_tensor /= 255.0
    
    
    img = cv2.imread("./please.png")

    # Resize image to 256x256 pixels
    img_resized = cv2.resize(img, (img_width, img_height))

    # Convert image to a PyTorch tensor with float values between 0 and 1
    img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1))).float() / 255.0


    # Add a batch dimension to the tensor
    random_image_tensor = img_tensor.unsqueeze(0)


    # cv2.imshow('Resized Image', img_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        
    return random_image_tensor


if __name__ == "__main__":
    
    training_set = H36_dataset(num_cams=num_cameras, subjectp=subjects[0:1], is_train = True) 
    test_set     = H36_dataset(num_cams=num_cameras, subjectp=subjects[0:1] , is_train = False)
    
    batch_size=64

    train_loader = DataLoader( training_set, shuffle=True, batch_size=batch_size, num_workers= 1)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=1)
    
    x = generate_image()
    
    d2d, d3d, frame = training_set.__getitem__(300)
    
    print(frame.shape)
    # frame =  torch.from_numpy(frame.transpose((2, 0, 1))).float() / 255.0
    frame =  torch.from_numpy(frame)
    frame = frame.unsqueeze(0)
    print(frame.shape)
    # breakpoint()

    model= Model_3D()
     
    # y= model(x)
    y2 = model(frame)
    
    # print(x.shape,y.shape)
    print(frame.shape,y2.shape)