Security Camera System
Introduction
Welcome to the Security Camera System project! Built using an NVIDIA Jetson Orin Nano, this application is designed to detect and track individuals, faces, and everyday objects (such as laptops and keyboards) in real time via a connected webcam stream.

This project was built to address real-world safety concerns by bringing proactive monitoring to spaces like classrooms, hallways, and private spaces. With a focus on situational awareness and user responsibility, this tool provides a practical, automated approach to safety—embracing a "better safe than sorry" mindset.

Demo & Previews
YouTube Demonstration: Watch the Project Demo

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

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Classify a live camera stream using an image recognition DNN.", 
    formatter_class=argparse.RawTextHelpFormatter, 
    epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage()
)

parser.add_argument("input", type=str, default="/dev/video0", nargs='?', help="URI of the input stream (e.g., /dev/video0)")
parser.add_argument("output", type=str, default="webrtc://@:8554/output", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="Pre-trained model to load (e.g. googlenet, resnet18)")
parser.add_argument("--topK", type=int, default=1, help="Show the topK number of class predictions (default: 1)")

try:
    args = parser.parse_known_args()[0]
except SystemExit:
    sys.exit(0)

# Load the image classification network
net = imageNet(args.network, sys.argv)

# Note: To load a custom-trained model, use:
# net = imageNet(model="model/resnet18.onnx", labels="model/labels.txt", 
#                input_blob="input_0", output_blob="output_0")

# Initialize video sources & outputs
input_stream = videoSource(args.input, argv=sys.argv)
output_stream = videoOutput(args.output, argv=sys.argv)
font = cudaFont()

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
