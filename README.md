# Object-Detection-in-games

Steps:
1. Install Visual C++ build tools
2. Download CUDA(10.1) and cuDNN(7.6)
3. Install CUDA
4. Copy the files from cuDNN to CUDA directory as follows :
		..\cuda\bin\cudnn64_7.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin
		..\cuda\include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include
		..\cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib

5. Clone this repo
6. Install Python 3.8
7. Make a new python virtual environment
8. Open Terminal in this environment and run "pip install -r requirement.txt"
9. Set the Game's display window to:
    - Window Mode: Windowed
    - Aspect ratio: 16:10
    - Resolution: 1280x800
    - Refresh rate: 60hz 
    - Preset settings: LOW

10. Run 'python main.py'

-Test images-

![ss1](https://user-images.githubusercontent.com/52231690/104280957-b5821a80-54d2-11eb-9763-4c7883e0d582.jpg)


![ss2](https://user-images.githubusercontent.com/52231690/104280978-bdda5580-54d2-11eb-9d40-231e2d0491a5.jpg)


