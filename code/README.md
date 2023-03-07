# Usage

1. Script 1: [test_on_image_mtcnn.py](test_on_image_mtcnn.py)
    
    This script uses the MTCNN face detector to detect faces in an image/folder of Images and then uses the Hopenet network to estimate the head pose of each face.
    
    ```bash
    python test_on_image_mtcnn.py [-h] [--gpu GPU_ID] --snapshot SNAPSHOT (--folder FOLDER | --image IMAGE)

    Head pose estimation using the Hopenet network.

    optional arguments:
    -h, --help           show this help message and exit
    --gpu GPU_ID         GPU device id to use. Default is [0]
    --snapshot SNAPSHOT  Path of model snapshot.
    --folder FOLDER      Folder containing images
    --image IMAGE        Image file to test

    ```

2. Script 2: [test_on_video_mtcnn.py](test_on_video_mtcnn.py)

    This script uses the MTCNN face detector to detect faces in a video and then uses the Hopenet network to estimate the head pose of each face.

    

