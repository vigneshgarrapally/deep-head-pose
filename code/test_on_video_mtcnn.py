#assumes that you are running the code from the code directory
import datetime
import sys, os, argparse
import cv2
import torch
import warnings
warnings.filterwarnings("ignore")
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import  hopenet, utils
from mtcnn import MTCNN

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='../models/hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--video', dest='video_path',type=str,help='Path of video file.If empty,uses webcam as input stream')
    parser.add_argument('--skip', dest='skip',type=int,help='Number of frames to skip between each detection. Considered for video inputs only',default=20)
    parser.add_argument('--save-log', help='whether to save logs', action='store_true',default=True)
    parser.add_argument('--live-demo', help='whether to run live demo', action='store_true',default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = Path('./outputs/videos')
    video_path = args.video_path
    if args.video_path is not None and not os.path.exists(args.video_path):
        sys.exit('Video does not exist. Please check the path again and try again.')
    if not Path(out_dir).exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    
    #if webcame, take filename as current time appended with webcam
    if args.video_path is None:
        filename=out_dir/Path('webcam'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        filename=out_dir/Path(args.video_path).stem
    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Initialize MTCNN face detector
    mtcnn_detector = MTCNN()

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print ('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    if video_path is None:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(filename)+'.avi', fourcc, 30.0, (width, height))

    if args.save_log:
        log_file=open(str(filename)+'.txt','w')
    frame_num=0
    while True:
        if video_path is not None and frame_num%args.skip!=0:
            frame_num+=1
            continue
        frame_num+=1
        ret,frame = video.read()
        if ret == False:
            break

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        #MTCNN face detection
        dets = mtcnn_detector.detect_faces(cv2_frame)

        #check if face is detected
        if len(dets) == 0:
            print("No face detected on frame: " + str(frame_num))
            continue
        #loop through all the faces detected
        for i in range(len(dets)):
            #get the bounding box
            x_min, y_min, width, height = dets[i]['box']
            #crop the image
            img = cv2_frame[y_min:y_min+height,x_min:x_min+width]
            #convert to PIL image
            img = Image.fromarray(img)
            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            img = Variable(img).cuda(gpu)

            yaw, pitch, roll = model(img)

            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data * idx_tensor) * 3 - 99

            if args.save_log:
                log_file.write('frame: %d, yaw: %f, pitch: %f, roll: %f' % (frame_num, yaw_predicted, pitch_predicted, roll_predicted))
            #draw axis on the image
            frame = utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_min+width)/2, tdy = (y_min + y_min+height)/2, size = height/2)
            #write the angles on the image using opencv
            cv2.putText(frame, "YAW: " + str(float(yaw_predicted)), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "PITCH: " + str(float(pitch_predicted)), (x_min, y_min+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "ROLL: " + str(float(roll_predicted)), (x_min, y_min+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if args.live_demo:
            #copy the frame
            frame_copy=frame.copy()
            #write text on the frame to show how to skip frames and exit
            cv2.putText(frame_copy, "Press n to skip to next frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame_copy, "Press q to exit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('frame',frame)
            key=cv2.waitKey(0) & 0xFF
            #if q is pressed, exit
            if key == ord('q'):
                break
            #if n is pressed, skip to next frame
            elif key == ord('n'):
                continue
        #write the frame
        out.write(frame)
        frame_num += 1
    if args.save_log:
        log_file.close()
    out.release()
    video.release()
