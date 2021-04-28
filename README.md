# Object-detection-in-games

Requirements
    
    PC Video game
    GPU with CUDA enabled


Steps

    Install Visual C++ build tools
    Download CUDA(10.1) and cuDNN(7.6)
    Install Python 3.8    
    Install CUDA
    Copy the files from cuDNN to CUDA directory as follows :
        ..\cuda\bin\cudnn64_7.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin
        ..\cuda\include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include
        ..\cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib
    Clone this repo
    Make a new python virtual environment
    Open Terminal in this environment and run "pip install -r requirements.txt"
    Set the Game's display window to:
     - Window Mode: Windowed
     - Aspect ratio: 16:10
     - Resolution: 1600:1200
     - Refresh rate: 60hz 
     - Preset settings: LOW
     - At top left corner    
     Run 'python main.py'


-Test images (from GTAV) -

![knf](https://user-images.githubusercontent.com/52231690/116390838-10e6d700-a83c-11eb-916e-d7ade18ee6dc.png)

![57fieb](https://user-images.githubusercontent.com/52231690/116390825-0debe680-a83c-11eb-85c5-216670861f17.gif)

