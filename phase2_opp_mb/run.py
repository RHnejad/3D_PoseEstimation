
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import subprocess


main_directory = "/Users/rh/test_dir/h3.6/dataset/my_videos/"


#coco2h36m:from https://github.com/Walter0807/MotionBERT/blob/main/lib/data/dataset_action.py with some changes 
def coco2h36m(x):
    '''
        Input: x (M x T x V x C) (changed, now is : (J,d)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[ 0,:] = (x[ 11,:] + x[ 12,:]) * 0.5
    y[ 1,:] = x[ 12,:]
    y[ 2,:] = x[ 14,:]
    y[ 3,:] = x[ 16,:]
    y[ 4,:] = x[ 11,:]
    y[ 5,:] = x[ 13,:]
    y[ 6,:] = x[ 15,:]
    y[ 8,:] = (x[ 5,:] + x[ 6,:]) * 0.5
    y[ 7,:] = (y[ 0,:] + y[ 8,:]) * 0.5
    y[ 9,:] = x[ 0,:]
    y[ 10,:] = (x[ 1,:] + x[ 2,:]) * 0.5
    y[ 11,:] = x[ 5,:]
    y[ 12,:] = x[ 7,:]
    y[ 13,:] = x[ 9,:]
    y[ 14,:] = x[ 6,:]
    y[ 15,:] = x[ 8,:]
    y[ 16,:] = x[ 10,:]
    return y

def save_to_json(video_name : list =None):
    
    # TEST = read_data()
    # T = 0

    outputs_directory = main_directory+"opp_outputs/"
    final_json_outputs_directory = os.path.join(main_directory,"final_json_outputs")
    
    videos = video_name if video_name else os.listdir(outputs_directory) 
    joints_for_json = list()  
    
    
    for video in videos:
        dir_name = video
        json_outputs_dir_path = os.path.join(outputs_directory, dir_name)
        jsons_force = os.path.join(json_outputs_dir_path, "jsons_force")

        if os.path.isdir(jsons_force):
            directory = os.listdir(jsons_force)
            directory.sort()
            for file_name in directory:
                if (file_name).endswith(".json"):
                    joint_with_conf = np.zeros((17,3))
                    with open(os.path.join(jsons_force,file_name)) as f:
                        data_json = json.load(f)
                        max_score_id, max_score = 0, 0
                        
                        for j in range(len(data_json)):
                            if data_json[j]['score'] > max_score:
                                max_score = data_json[j]['score']
                                max_score_id = j
                                
                        if len(data_json) > 0 :
                            joint_with_conf = np.array(data_json[max_score_id]["keypoints"]).reshape(17,3)
                            joint_with_conf[:,:2] = coco2h36m(joint_with_conf[:,:2])
                            # joint_with_conf[:,:2] = joint_with_conf[:,:2]/
                        else:
                            print(data_json, file_name)
                            
                    loop_temp_dict =  { "image_id": file_name, "category_id": 1, "keypoints":joint_with_conf.tolist() , "score":max_score  }
                    # loop_temp_dict =  { "image_id": file_name, "category_id": 1, "keypoints":TEST[T*5].tolist() , "score":max_score  }
                    # T = T+1

                    joints_for_json.append(loop_temp_dict)
                    
                # print(joints_for_json)    
                
            with open(os.path.join(final_json_outputs_directory,video+'.json'), 'w') as handle:    
                json.dump(joints_for_json,handle)
        else:
            print(f"{jsons_force} not a directroy")

       
def run_ffmpeg(single_video:list=None, fps=10 ):
    video_directory = main_directory+"raw_videos/"
    frame_directory = main_directory+"ffmpeg_frames/"
    reduced_fps_videos_directory =  main_directory+"reduced_fps_videos/"

    videos = single_video if single_video else os.listdir(video_directory)
    print(videos)

    for video in videos:
        dir_name = video
        dir_path = os.path.join(frame_directory, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if video.endswith(".mp4"): 
            input_file = os.path.join(video_directory, video)
            output_file = os.path.join(dir_path, '%04d.jpg')
            os.system("/Users/rh/audio-orchestrator-ffmpeg/bin/ffmpeg  -i '"+str(input_file)+ f"'  -vf fps={fps} '" + str(output_file)+ "'")
            output_file = os.path.join(reduced_fps_videos_directory,video+"_fps.mp4")
            os.system("/Users/rh/audio-orchestrator-ffmpeg/bin/ffmpeg  -i '"+str(input_file)+ f"'   -vf fps={fps}  '" + str(output_file)+ "'")
 
            
def run_openpifpaf(single_video:list=None ):
    
    outputs_directory = main_directory+"opp_outputs/"
    all_frames_directory = main_directory+"ffmpeg_frames/"
    
    folders = single_video if single_video else os.listdir(all_frames_directory)
    
    for video in folders:
        print(video)
        dir_name = video
        json_output_dir_path = os.path.join(outputs_directory, dir_name)
        jsons_force = os.path.join(json_output_dir_path, "jsons_force")
        imageOuts_force = os.path.join(json_output_dir_path, "imgOuts_force")
        if not os.path.exists(json_output_dir_path):
            os.makedirs(json_output_dir_path)
            os.makedirs(jsons_force)
            os.makedirs(imageOuts_force)
        
        frames_directory = os.path.join(all_frames_directory,video)
        if os.path.isdir(frames_directory):
            for frame in os.listdir(frames_directory):
                frame_directory = os.path.join(frames_directory,frame)
                print("FRAME:",frame)
                if frame.endswith(".jpg"):
                    
                    bash_command = "python3 -m openpifpaf.predict '"+str(frame_directory)+ \
                    "' --checkpoint shufflenetv2k30 --force-complete-pose --instance-threshold 0.2" + \
                    " --json-output '"+str(jsons_force) +"'" #+ "--image-output '"+ str(imageOuts_force) 
                    #seed treshhold / 
                    
                    subprocess.run(bash_command, shell=True)
        else :
            print(f"{frames_directory} not a directroy")

 
# source /Users/rh/test_dir/env1/bin/activate
# cd /Users/rh/test_dir/h3.6/dataset/trimmed_fps7/
# mkdir jsons_force
# mkdir imgOuts_force
# for file in ./*.jpg;do
# 	python3 -m openpifpaf.predict   "$file" --json-output  /Users/rh/test_dir/h3.6/dataset/trimmed_fps7/jsons_force/ --image-output /Users/rh/test_dir/h3.6/dataset/trimmed_fps7/imgOuts_force/ --force-complete-pose
# done
 
def read_data(action = "Walking 1"): #ch

    cam_ids = [".54138969", ".55011271", ".58860488",  ".60457274", ]

    path_positions_3d_VD3d =  "/Users/rh/test_dir/h3.6/VideoPose3D/data/npz/" + "data_3d_h36m.npz"
    data_file_3d = np.load(path_positions_3d_VD3d, allow_pickle=True)

    data_file_3d = data_file_3d['positions_3d'].item()
    KeyPoints_from3d = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]

    n_frame = 0 
    for s in ["S1"]:
        for a in data_file_3d[s].keys():
            if (action == a ) :
                n_frame += len(data_file_3d[s][a])  
         
    all_in_one_dataset_3d = np.zeros((n_frame, 17 ,3),  dtype=np.float32)
    
    i = 0
    for s in ["S1"]:
        for a in data_file_3d[s].keys():
            if (action == a ):
                print(s,a,len(data_file_3d[s][a]))
                for frame_num in range(len(data_file_3d[s][a])):

                    global_pose = data_file_3d[s][a][frame_num]  
                    global_pose = global_pose[ KeyPoints_from3d ,:]
                    tmp = global_pose.copy()
                    # for j in range(len(tmp)): 
                    #     tmp[j] = tmp[j] - np.divide(np.array(camera_parameters[s][3]['translation']),1000)
                    #     tmp[j] = qv_mult(np.array(camera_parameters[s][3]['orientation']),tmp[j])

                    all_in_one_dataset_3d[i] = tmp
                    i = i+1
    for i in range(n_frame):
        all_in_one_dataset_3d[i,1:] = all_in_one_dataset_3d[i,1:] - all_in_one_dataset_3d[i,0]      
        all_in_one_dataset_3d[i,0] *= 0          
    print(n_frame)
    return all_in_one_dataset_3d 



def visualize_2d(keypoints, frame=None, name = "kp"):
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    plt.figure()

    plt.imshow(frame) #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for i in range(17):
        plt.plot(keypoints.T[0][sk_points[i]], keypoints.T[1][sk_points[i]], "y" )
    plt.plot(keypoints.T[0],keypoints.T[1], "ob", markersize=4)

    # plt.xlim([0,620])
    # plt.ylim([380,0])
    
    plt.savefig(name)
    plt.close()

def visualize_3d(keypoints,keypoints2, name="3d"):
    sk_points = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]

    if keypoints.shape[0] != 17 :
        keypoints = np.insert(keypoints , 0 , values= [0,0,0], axis=0 )
        keypoints2 = np.insert(keypoints2 , 0 , values= [0,0,0], axis=0 )

    plt.figure()
    ax = plt.axes(projection='3d')

    xdata = keypoints.T[0]
    ydata = keypoints.T[1]
    zdata = keypoints.T[2]
    ax.scatter(xdata,ydata,zdata,"b",label="gt")
    for i in range(17):
        ax.plot(xdata[sk_points[i]], ydata[sk_points[i]], zdata[sk_points[i]] , "b" )

    xdata2 = keypoints2.T[0]
    ydata2 = keypoints2.T[1]
    zdata2 = keypoints2.T[2]
    ax.scatter(xdata2,ydata2,zdata2, "r" , label ="estimation")
    for i in range(17):
        ax.plot(xdata2[sk_points[i]], ydata2[sk_points[i]], zdata2[sk_points[i]] , "r" )

    plt.legend()

    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1 , top=1 ) 

    plt.savefig(name) #ch
    plt.close()
    
    
    
def create_2d_mp4(video=None, fps=10):
    
    final_json_outputs_directory = os.path.join(main_directory,"final_json_outputs")
    all_frames_directory = main_directory+"ffmpeg_frames/" 
    opp_2d_frames_directory = main_directory+"opp_2d_frames/"
    
    frames_directory = os.path.join(all_frames_directory,video)
    opp_2d_frames_vid = os.path.join(opp_2d_frames_directory,video)
    
    if not os.path.exists(opp_2d_frames_vid):
        os.makedirs(opp_2d_frames_vid)
    
    with open(os.path.join(final_json_outputs_directory,video+'.json')) as f: #ch
        data = json.load(f)
   
    i =1
    for datum in data:                    
        keypoints=np.array(datum["keypoints"])[:,:2]
        frame=cv2.imread(os.path.join(frames_directory,str(i).zfill(4)+".jpg"))
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)
        visualize_2d(keypoints=keypoints,frame=frame,name=os.path.join(opp_2d_frames_vid,str(i).zfill(4)+'.png')) 
        if i%10 == 0:
            print(i)
        i += 1
    
    bash_command = f"/Users/rh/audio-orchestrator-ffmpeg/bin/ffmpeg -framerate {fps} -pattern_type glob -i '"\
        +opp_2d_frames_vid+"/*.png' -c:v libx264 -pix_fmt yuv420p  '"+opp_2d_frames_vid+"/out.mp4'"   
    subprocess.run(bash_command, shell=True)
    
    for file in os.listdir(opp_2d_frames_vid):
        if file.endswith('.png'):
            os.remove(os.path.join(opp_2d_frames_vid,file)) 
    
def create_3d_mp4(video=None, fps=10):
    
    import sys
    sys.path.append("/Users/rh/test_dir/h3.6/AE/")
    from utils import camera_parameters
    q_cam = np.array(camera_parameters["S1"][2]['orientation']) #ch
    t_cam = np.divide(np.array(camera_parameters["S1"][2]['translation']),1000) #ch
    R_cam = np.array([
    [1 - 2*q_cam[2]**2 - 2*q_cam[3]**2, 2*q_cam[1]*q_cam[2] - 2*q_cam[0]*q_cam[3], 2*q_cam[0]*q_cam[2] + 2*q_cam[1]*q_cam[3]],
    [2*q_cam[1]*q_cam[2] + 2*q_cam[0]*q_cam[3], 1 - 2*q_cam[1]**2 - 2*q_cam[3]**2, 2*q_cam[2]*q_cam[3] - 2*q_cam[0]*q_cam[1]],
    [2*q_cam[1]*q_cam[3] - 2*q_cam[0]*q_cam[2], 2*q_cam[0]*q_cam[1] + 2*q_cam[2]*q_cam[3], 1 - 2*q_cam[1]**2 - 2*q_cam[2]**2]
    ])
    
    #___
    
    MB_npy_outputs_directory = os.path.join(main_directory,"MB_npy")
    MB_3d_frames_directory = main_directory+"MB_3d_frames/"
    
    frames_directory = os.path.join(MB_3d_frames_directory,video)
    
    if not os.path.exists(frames_directory):
        os.makedirs(frames_directory)
    

    data_mb = np.load(os.path.join(MB_npy_outputs_directory,"yuzu_black.mp4"+".npy")) #ch
    data_gt= read_data("Walking 1")

    # for i in range(data_mb.shape[0]):
    #     data_mb[i] += t_cam

    data_mb = np.matmul(data_mb, R_cam.T) #global
   
    # for i in range(data_mb.shape[0]):   
    #     data_mb[i,1:] = data_mb[i,1:] - data_mb[i,0]      
    #     data_mb[i,0] *= 0   
    
    for i in range(data_mb.shape[0]):
        visualize_3d(data_gt[i*5*0].copy()*0,2.8*data_mb[i],name=os.path.join(frames_directory,str(i).zfill(4)+'.png') ) #ch
        if i%10==0:
            print(i)
    
    bash_command = f"/Users/rh/audio-orchestrator-ffmpeg/bin/ffmpeg -framerate {fps} -pattern_type glob -i '"\
        +frames_directory+"/*.png' -c:v libx264 -pix_fmt yuv420p  '"+frames_directory+"/out.mp4'"   
    subprocess.run(bash_command, shell=True)
    
    for file in os.listdir(frames_directory):
        if file.endswith('.png'):
            os.remove(os.path.join(frames_directory,file)) 
            
                
if __name__ == "__main__":
    
    video = "Walking 1.58860488.mp4"
    video = "yuzu_black.mp4"
    fps = 10
    # run_ffmpeg([video], fps=fps)
    # run_openpifpaf([video])
    # save_to_json([video])  
    # create_2d_mp4(video, fps=fps)
    create_3d_mp4("Walking 1", fps=fps)
    print("___DONE___")
    
    
    
# python3 infer_wild.py \
# --vid_path /home/rh/codes/MotionBERT/my_videos/trimmed_fall.mp4 \
# --json_path /home/rh/codes/MotionBERT/my_videos/json_force_fall.json \
# --out_path /home/rh/codes/MotionBERT/my_infered_outputs