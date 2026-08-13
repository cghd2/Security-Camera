**Security Camera System**
**Introduction**
Welcome to the Security Camera System project! Built using an NVIDIA Jetson Orin Nano, this application is designed to detect and track individuals, faces, and everyday objects (such as laptops and keyboards) in real time via a connected webcam stream.

This project was built to address real-world safety concerns by bringing proactive monitoring to spaces like classrooms, hallways, and private spaces. With a focus on situational awareness and user responsibility, this tool provides a practical, automated approach to safety—embracing a "better safe than sorry" mindset.

Visuals
How It Works
This project is built using Python inside Visual Studio Code and leverages deep learning models provided by NVIDIA's jetson-inference library:

DetectNet / PoseNet: Used for real-time human detection, pose estimation, and bounding-box spatial tracking.

ImageNet: Utilized for multi-class object detection and classification.

Custom Models: Trained with custom dataset images for tailored object/subject classification on the Jetson Orin Nano.

Installation
Ensure you have the jetson-inference library installed on your NVIDIA Jetson device alongside standard Python dependencies.

Bash
# Clone the jetson-inference repository
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build && cd build
cmake ../
make
sudo make install
sudo ldconfig
How To Run
1. Direct Command Line
To run object detection directly using the C++/Python utility:

Bash
detectnet /dev/video0 webrtc://@:8554/output
2. Python Script (imagenet.py)
Run the expanded Python classification script below with your choice of network and live video streams:

Bash
python3 main.py /dev/video0 webrtc://@:8554/output --network=googlenet
Python Source Code (main.py)
Python
#!/usr/bin/env python3

import sys
import argparse

from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log


# Process video frames continuously
while True:
    # Capture the next image frame
    img = input_stream.Capture()

    if img is None: # Timeout or stream interrupted
        continue  

    # Classify image frame and fetch top-K predictions
    predictions = net.Classify(img, topK=args.topK)

    # Render label overlays on screen
    for n, (classID, confidence) in enumerate(predictions):
        classLabel = net.GetClassLabel(classID)
        confidence_pct = confidence * 100.0

        print(f"ImageNet: {confidence_pct:05.2f}% class #{classID} ({classLabel})")

        font.OverlayText(
            img, 
            text=f"{confidence_pct:05.2f}% {classLabel}", 
            x=5, 
            y=5 + n * (font.GetSize() + 5),
            color=font.White, 
            background=font.Gray40
        )
                           
    # Output the frame to the network stream
    output_stream.Render(img)

    # Display performance metrics on title bar
    output_stream.SetStatus(f"{net.GetNetworkName()} | {net.GetNetworkFPS():.0f} FPS")

    # Log profiler timing data
    net.PrintProfilerTimes()

    # Exit loop if stream ends
    if not input_stream.IsStreaming() or not output_stream.IsStreaming():
        break
By the way, to unlock the full functionality of all Apps, enable Gemini Apps Activity.


**YouTube Demo:
**

https://www.youtube.com/watch?v=RSswoXz1cLM


**Screenshots**

<img width="818" height="745" alt="Screenshot 2026-08-13 102132" src="https://github.com/user-attachments/assets/1746f4b1-161b-411d-875b-f22db36a94c4" />
<img width="525" height="66" alt="Screenshot 2026-08-13 102042" src="https://github.com/user-attachments/assets/bb43e8f3-25cc-4bf6-be9c-b00e029dc8ac" />

