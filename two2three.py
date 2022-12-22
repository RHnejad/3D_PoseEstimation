



def load_data(data_path, dim=2, standardize=True, subjects = subjects, sample = sample, Samples = Samples, zero_centre=zero_centre):

    data_file = np.load(data_path, allow_pickle=True)
    num_of_joints = 17 + (dim-2)*15 # 17 or 32

    data_file = data_file['positions_'+str(dim)+'d'].item()


    #zero centering around the root
    if zero_centre :
        for s in subjects:
            for a in data_file[s].keys():
                if dim == 2 :
                    for cam in range(4):
                        for frame in range(len(data_file[s][a][cam])) :
                            data_file[s][a][cam][frame][1:] -= data_file[s][a][cam][frame][:1]

                elif dim == 3 :
                    for frame in range(len(data_file[s][a])):
                        data_file[s][a][frame][1:] -= data_file[s][a][frame][:1]

    frames_count = 0
    #calculating mean and counting num of frames 
    data_sum =np.zeros((num_of_joints ,dim))
    for s in subjects:
        for a in data_file[s].keys():
            if dim == 2 :
                for cam in range(4):
                    frames_count += len(data_file[s][a][cam])
                    for frame in range(len(data_file[s][a][cam])) :
                        data_sum += data_file[s][a][cam][frame]
                mean_of_all_frames = np.divide(data_sum,frames_count)

            elif dim == 3 :
                frames_count += len(data_file[s][a])
                for frame in range(len(data_file[s][a])):
                    data_sum += data_file[s][a][frame]
                mean_of_all_frames = np.divide(data_sum,frames_count)


    #calculating std 
    diff_sq2_sum =np.zeros((num_of_joints,dim))
    for s in subjects:
        for a in data_file[s].keys():
            if dim == 2:
                for cam in range(4):
                    for frame in range(len(data_file[s][a][cam])) :
                        diff_sq2_sum += ((data_file[s][a][cam][frame] - mean_of_all_frames)**2)
                std_dev = np.sqrt(np.divide(diff_sq2_sum,(frames_count-1)))

            elif dim ==3 :
                for frame in range(len(data_file[s][a])) :
                    diff_sq2_sum += ((data_file[s][a][frame] - mean_of_all_frames)**2)
                std_dev = np.sqrt(np.divide(diff_sq2_sum,(frames_count-1)))

    #saving everything in one np array
    all_in_one_dataset = np.zeros((frames_count,num_of_joints,dim))
    i=0
    
    for s in subjects:
        for a in data_file[s].keys():
            if dim == 2 :
                for cam in range(4):
                    for frame in range(len(data_file[s][a][cam])) :
                        all_in_one_dataset[i] = np.divide(data_file[s][a][cam][frame] - mean_of_all_frames, std_dev) if standardize else data_file[s][a][cam][frame] #also standardize (x-mean_2d)/std_2d
                        if zero_centre : all_in_one_dataset[i][0] *= 0
                        i+=1
            if dim == 3 :
                for frame in range(len(data_file[s][a])) :
                    all_in_one_dataset[i] =  np.divide(data_file[s][a][frame] - mean_of_all_frames, std_dev) if standardize else data_file[s][a][frame]
                    if zero_centre : all_in_one_dataset[i][0] *= 0
                    i+=1

    if dim == 3 : 
        all_in_one_dataset = np.delete (all_in_one_dataset,KeyPoints_17_from3d_to_delete , axis=1)
    
    if dim == 2 and sample :
        all_in_one_dataset = all_in_one_dataset.reshape((int(frames_count/4),4, 17,2))
    
    # S , L= 10000 , 100
    # returned_data = all_in_one_dataset[ (3*(dim-2)+1)*S  : (3*(dim-2)+1)*S  + (-3*dim+10)*L   ]
    
    returned_data = all_in_one_dataset[Samples] if  sample else all_in_one_dataset
    if dim == 2 and sample :
        returned_data = returned_data.reshape(-1, 17,2)

    # print(subjects,returned_data)    

    return returned_data, mean_of_all_frames , std_dev #[:(-3*dim+10)*100] #all_in_one_dataset[(-3*dim+10)*89900:(-3*dim+10)*90000]  pay attention taht it's divisable by 4 


def visualize(keypoints,st_kp, mean , std, name = "kp", ):

    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    if not IZAR :
        import cv2
        cap= cv2.VideoCapture('/Users/rh/test_dir/h3.6/dataset/Walking.54138969.mp4')
        i=0
        while(cap.isOpened()) and i<=0 :
            ret, frame = cap.read()
            i+=1
        cap.release()
        cv2.destroyAllWindows()

    plt.figure()
    if not IZAR : plt.imshow(frame)
    u,v = keypoints.T[0],keypoints.T[1]
    u,v = u*std[:,0]+mean[:,0], v*std[:,1]+mean[:,1]
    plt.plot(u,v, "ob", markersize=4)
    for i in range(17):
        plt.plot(u[sk_points[i]], v[sk_points[i]] , "b" )

    u,v = st_kp.T[0],st_kp.T[1]
    u,v = u*std[:,0]+mean[:,0], v*std[:,1]+mean[:,1]  
    plt.plot(u,v, "or", markersize=3)
    for i in range(17):
        plt.plot(u[sk_points[i]], v[sk_points[i]] , "r" )
    plt.xlim([0,1e3]), plt.ylim([1e3,0])

    # plt.show()
    # if IZAR : plt.savefig(name +'.png')
    plt.savefig("pics/"+name +'.png')
    plt.show()

def visualize_3d(keypoints,keypoints2, name="3d"):
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    plt.figure()
    ax = plt.axes(projection='3d')

    xdata = keypoints.T[0]
    ydata = keypoints.T[1]
    zdata = keypoints.T[2]
    ax.scatter(xdata,ydata,zdata,"b",label="expectations")
    for i in range(17):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], zdata[sk_points[i]] , "b" )

    xdata2 = keypoints2.T[0]
    ydata2 = keypoints2.T[1]
    zdata2 = keypoints2.T[2]
    ax.scatter(xdata2,ydata2,zdata2, "r" , label ="reality")
    for i in range(17):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], zdata2[sk_points[i]] , "r" )

    plt.legend()

    ax.axes.set_xlim3d(left=-2, right=2) 
    ax.axes.set_ylim3d(bottom=-2, top=2) 
    ax.axes.set_zlim3d(bottom=0, top=2) 
    plt.savefig("pics/"+name +'.png')
    plt.show()

#________

class Pose_KeyPoints(Dataset):
    def __init__(self, num_cams = 4, subject=subjects , transform=None, target_transform=None):

        self.dataset2d,self.mean_2d,self.std_2d = load_data(data_path=path_positions_2d_VD3d, dim=2, subjects=subject, sample = False if len(subject)==2 else sample)
        self.dataset3d,self.mean_3d,self.std_3d = load_data(data_path=path_positions_3d_VD3d, dim=3, subjects=subject, sample = False if len(subject)==2 else sample, standardize=False)
        self.transform = transform
        self.target_transform = target_transform
        self.num_cams = num_cams

        self.encoding = (np.divide (np.array(range(-8,9)), 18)).reshape(17,1)

        print ("///////",(self.dataset3d).shape, (self.dataset2d).shape)

    def __len__(self):
        return len(self.dataset3d) #number of all the frames 

    def __getitem__(self, idx):
        return self.dataset2d[idx*4 :idx*4+self.num_cams].reshape(-1,2), self.dataset3d[idx] #cam 0 
#______
class AE(nn.Module):
    def __init__(self, input_dim=2, output_dim=3):
        super().__init__()
         
        self.input_dim = input_dim *17
        self.output_dim = output_dim *17

        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 12),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(12, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.output_dim )         
        )

        self.acti_final = torch.nn.Tanh()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if self.output_dim == 2 :
            decoded = self.acti_final(decoded)
        return decoded
#______

def get_positional_embeddings(sequence_length=17, d=2):
    result = torch.ones(sequence_length,d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i/((1e4)**(j/d))) if j%2==0 else np.cos(i/((1e4)**((j-1)/d)))
    return result

class MyMSA(nn.Module): #MultiHeadSelfAttention
    def __init__(self, d=2, n_heads =1):
        super(MyMSA, self).__init__()
        self.d = d 
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimention {d} into {n_heads} heads"

        d_head = int(d/n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range (self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range (self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range (self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim) (N,17,2)
        # We got into shape (N, seq_len, n_heads, token_dim / n_heads)  (N,17,1,2)
        # And come back to (N, seq_len, item_dim) (through concatenation)
        result = []
        for sequence in sequences :
            seq_result = []
            for head in range(self. n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence [:, head*self.d_head:(head+1)*self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
            return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

class MyViTBlock(nn.Module):
  def __init__(self, hidden_d, n_heads, mlp_ratio=4):
    super(MyViTBlock, self).__init__()
    self.hidden_d = hidden_d
    self.n_heads = n_heads

    self.norm1 = nn.LayerNorm(hidden_d)
    self.mhsa = MyMSA(hidden_d, n_heads)
    self.norm2 = nn.LayerNorm(hidden_d)
    self.mlp = nn.Sequential(
        nn.Linear(hidden_d, mlp_ratio*hidden_d),
        nn.GELU(), 
        nn.Linear(mlp_ratio * hidden_d, hidden_d )
    )

  def forward(self, x):
    out = x + self.mhsa(self.norm1(x))
    out = out + self.mlp(self.norm2(out))
    return out

class MyViT(nn.Module):
  def __init__(self, chw=(1,17,2),  n_blocks=2, hidden_d=8, n_heads=2, out_d=(17*3)):
    #super constructer
    super(MyViT,self).__init__()

    #attributes
    self.chw = chw #(C,H,W)
    # self.n_patches = n_patches
    self.n_block = n_blocks
    self.n_heads = n_heads
    self.hidden_d = hidden_d

    # #Input and patches sizes
    # assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number if patches"
    # assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number if patches"
    # self.patch_size = (chw[1]/n_patches, chw[2]/n_patches) -> (28/7, 28/7) (4,4)

    # 1) Linear mapper 
    self.input_d = int(1*2) #int(chw[0]*self.patch_size[0]*self.patch_size[1]) (4*4=16)
    self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

    # 2) Learnable classification token 
    self.class_token = nn.Parameter(torch.rand(1,self.hidden_d))

    # 3) Positional embedding
    self.pos_embed = nn.Parameter(torch.tensor(get_positional_embeddings(chw[1]+1,self.hidden_d))) #why the plus one???????
    self.pos_embed.requires_grad = False

    # 4) Transformer encoder blocks
    self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]) 

    # 5) Classification MLPK
    self.mlp = nn.Sequential(
        nn.Linear(self.hidden_d, 16), # 17*4=68
        torch.nn.ReLU(), 
        nn.Linear(16, 32),
        torch.nn.ReLU(),
        nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, out_d)#, #17*3=51
        # nn.Tanh()#(dim=-1)
    )

  def forward(self,images):
    #Divising images into patches
    n, h, w = images.shape #n,c, h, w
    patches = images # patchify(images, self.n_patches)

    #Running linear layer tokenization 
    #Map the vector corresponding to each path to the hidden size dimension 
    tokens = self.linear_mapper(patches)

    #Adding colassification token to the tokens 
    tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

    #Adding positional embedding 
    pos_embed = self.pos_embed.repeat(n,1,1)
    out = tokens + pos_embed 

    #Transformer Blocks 
    for block in self.blocks: 
      out = block(out)

    # print("*",out)

    # Getting the classification token only 
    out = out [:, 0]

    # print(out)

    return self.mlp(out) #Map to output dimention, output category distribution


#______

def loss_MPJPE(prediction, target):
    # print("in loss" ,target.shape, len(target.shape)-1)
    loss = torch.mean(torch.norm(prediction-target,dim=len(target.shape)-1) )
    return loss

def main():

    training_set = Pose_KeyPoints(num_cams=num_cameras, subject=subjects[:5])
    test_set = Pose_KeyPoints(num_cams=num_cameras, subject=subjects[5:7])
    # print(training_set.mean_2d)
    # print(training_set.mean_3d)
    train_loader = DataLoader( training_set, shuffle=True, batch_size=2048)
    test_loader = DataLoader( test_set, shuffle=True, batch_size=2048)
    print("data loader")

    #Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # model = MyViT((1,17*num_cameras,2),n_blocks=2 , hidden_d=2, n_heads=1, out_d=(17*output_dimention)).to(device)   #, n_patches=7, n_blocks=2, 
    # model = TransformerAE(2*num_cams, 3, 0.2).to(device)
    model = AE(input_dimention, output_dimention).to(device)

    n_epochs=1000
    lr = 1e-3

    #Traning loop
    loss_function = torch.nn.MSELoss()
    # loss_function = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(),
                             lr = lr,
                             weight_decay = 1e-8) #1e-8

    flag = 1
    for epoch in tqdm(range(n_epochs),desc="Training"):
        train_loss = 0.0
        batch_loss = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} in training", leave=False):

            x, y  = batch
            x,y=x.float(),y.float()

            x, y = x.to(device), y.to(device)

            y = x if (output_dimention == 2) else y

            optimizer.zero_grad()

            y_hat = model(x)

            if flag :
                    ytmp = y.cpu().detach().numpy().reshape(-1, 17,output_dimention)
                    y_hattmp = y_hat.cpu().detach().numpy().reshape(-1, 17,output_dimention) 
                    print(y_hattmp.shape, ytmp.shape) 
                    if output_dimention == 3 :
                        visualize_3d(ytmp[0],y_hattmp[0],"y3dflag_b")
                    else :
                        visualize(ytmp[0],y_hattmp[0], training_set.mean_2d , training_set.std_2d ,"y2dflag")
                    flag = 0

            y_hat = y_hat.reshape(-1,17,output_dimention)
            # loss = criterion (y_hat, y) new
            loss = loss_function(y_hat, y) 
            # loss = loss_MPJPE(y_hat, y)

            batch_loss.append(loss.detach().cpu().item())
            train_loss += loss.detach().cpu() / len(train_loader)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('Batch Loss: {}'.format(batch_loss))
        print(f"epoch {epoch+1}/{n_epochs} loss: {train_loss:.2f} , "+str(loss_MPJPE(y_hat, y))) 

    x = x.cpu().detach().numpy().reshape(-1, 17,int(input_dimention/num_cameras))
    x2= np.multiply(x[0],training_set.std_2d)+training_set.mean_2d

    y = y.cpu().detach().numpy().reshape(-1, 17,output_dimention)
    y_hat = y_hat.cpu().detach().numpy().reshape(-1, 17,output_dimention) 

    if output_dimention == 3 :
        visualize_3d(y[0],y_hat[0],"y2_3d4")
        visualize_3d(y[-1],y_hat[-1],"y2b_3d4")
    else :
        visualize(y[0],y_hat[0], training_set.mean_2d , training_set.std_2d, "y2_2d_b")
        visualize(y[-1],y_hat[-1], training_set.mean_2d , training_set.std_2d, "y2b_2d_b")      

    print(y.shape, y_hat.shape)


    with torch.no_grad():
        model.eval()
        total_loss = 0
        test_loss = 0.0
        for x, y in test_loader:
            x,y=x.float(),y.float()
            x, y = x.to(device), y.to(device)
            y = x if (output_dimention == 2) else y

            y_hat = model(x)
            y_hat = y_hat.reshape(-1,17,output_dimention)       
            # loss = loss_function(y_hat, y) 
            loss = loss_MPJPE(y_hat, y)

            test_loss += loss.detach().cpu() / len(train_loader)

        print(f"TEST loss: {test_loss:.2f} , ") 

    x = x.cpu().detach().numpy().reshape(-1, 17,int(input_dimention/num_cameras))
    x2= np.multiply(x[0],training_set.std_2d)+training_set.mean_2d

    y = y.cpu().detach().numpy().reshape(-1, 17,output_dimention)
    y_hat = y_hat.cpu().detach().numpy().reshape(-1, 17,output_dimention) 

    if output_dimention == 3 :
        visualize_3d(y[0],y_hat[0],"test_3d4")
        visualize_3d(y[-1],y_hat[-1],"test2_3d4")
    else :
        visualize(y[0],y_hat[0], training_set.mean_2d , training_set.std_2d, "test_2d")
        visualize(y[-1],y_hat[-1], training_set.mean_2d , training_set.std_2d, "test2_2d")     

if __name__ == "__main__" :
    print("start")
    # test,mean2d,std2d = load_data(path_positions_2d_VD3d) 
    # test3d,mean_3d,std_3d = load_data(path_positions_3d_VD3d,dim=3) 
    main()
