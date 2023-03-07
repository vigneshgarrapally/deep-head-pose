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

    This script uses the MTCNN face detector to detect faces in a video/webcam stream and then uses the Hopenet network to estimate the head pose of each face.

    ```bash
    python test_on_video_mtcnn.py [-h] [--gpu GPU_ID] [--snapshot SNAPSHOT] [--video VIDEO_PATH] [--skip SKIP] [--save-log] [--live-demo]

    Head pose estimation using the Hopenet network.

    optional arguments:
    -h, --help           show this help message and exit
    --gpu GPU_ID         GPU device id to use [0]
    --snapshot SNAPSHOT  Path of model snapshot.
    --video VIDEO_PATH   Path of video file.If empty,uses webcam as input stream
    --skip SKIP          Number of frames to skip between each detection. Considered for video inputs only
    --save-log           whether to save logs
    --live-demo          whether to run live demo
    ```

    Example usage:
    ```bash
        python .\code\test_on_video_mtcnn.py --snapshot .\models\hopenet_robust_alpha1.pkl --live-demo
    ```
    > **_NOTE:_**  Run Scripts from root directory of the repository instead of code directory to avoid any errors with respect to file paths.

    

