import sys, os, argparse
import cv2

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import hopenet, utils
from mtcnn import MTCNN




if __name__ == '__main__':
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use. Default is [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          required=True, type=str)
    input_group=parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--folder',help='Folder containing images',type=str)
    input_group.add_argument('--image', help='Image file to test',type=str)
    args = parser.parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = Path('outputs/images')
    #check if snapshot exists and is file
    if not Path(snapshot_path).exists() or not Path(snapshot_path).is_file():
        print("Snapshot does not exist")
        sys.exit()
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True)
    
    #check if folder or image is provided
    if args.folder:
        folder_path = args.folder
        if not Path(folder_path).exists() or not Path(folder_path).is_dir():
            print("Folder does not exist")
            sys.exit()
    elif args.image:
        image_path = args.image
        if not Path(image_path).exists() or not Path(image_path).is_file():
            print("Image does not exist")
            sys.exit()
    print('Loading hopenet.')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
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
    if args.folder:
        files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    elif args.image:
        files = [args.image]
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