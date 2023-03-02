import sys, os, argparse
import cv2

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import hopenet, utils
from mtcnn import MTCNN




def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='', type=str)
    parser.add_argument('--folder', dest='folder', help='Folder containing images')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/images'
    folder_path = args.folder

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.folder):
        sys.exit('Folder does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

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
    #get all the .png files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    frame_num = 1
    #initialize mtcnn detector
    detector = MTCNN()
    for file in files:
        # Read image
        print("Processing Frame: " + str(frame_num))
        save_path=out_dir + '/' + file
        frame=cv2.imread(folder_path + file)
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # MTCNN face detection
        dets = detector.detect_faces(cv2_frame)
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
            #draw axis on the image
            frame = utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_min+width)/2, tdy = (y_min + y_min+height)/2, size = height/2)
            #write the angles on the image using opencv
            cv2.putText(frame, "YAW: " + str(yaw_predicted), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "PITCH: " + str(pitch_predicted), (x_min, y_min+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "ROLL: " + str(roll_predicted), (x_min, y_min+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #save the image
            print(save_path)
            cv2.imwrite("output/images/" + str(frame_num) + ".jpg", frame)
        frame_num += 1