# OSSDC-VisionBasedACC
Discuss requirments and develop code for #1-mvp-vbacc MVP (see also this channel on ossdc.org Slack)

We have two initial demos videos here, showing how detection/tracking algoritms can be used:

- Update demo code is here (see run demo scripts):
  - https://github.com/OSSDC/OSSDC-VisionBasedACC/tree/master/object_detection 

- [SSD Tensorflow based car detection and tracking demo for OSSDC.org VisionBasedACC PS3/PS4 simulator ](https://www.youtube.com/watch?v=dqnjHqwP68Y)
  - The code for the demo is here (see ossdc scripts): 
    - https://github.com/OSSDC/SSD-Tensorflow/tree/master/notebooks
  - It can ran in real time on local videos or network streams, including directly on YouTube videos (using youtube-dl program), like I described here:
    - https://github.com/OSSDC/SSD-Tensorflow/commit/26fa7eea5155ac3989476936628e62be3d773b95

- [Real time YOLO detection in OSSDC Simulator running TheCrew on PS4 30fps](https://www.youtube.com/watch?v=ANgDlNfDoAQ)
  - The code for the demo is here: 
    - https://github.com/OSSDC/darknet
  - See here instructions on how to run it:
    - https://github.com/OSSDC/darknet/commit/dc76e5690c753ce093417e096313b390b5d5c2ae

It can be noted in the videos above that SSD Tensorflow is more accurate than the Yolo version.

Here are a few articles with more details about VBACC and OSSDC:
  - How OSSDC was born: 
    - [SDC From 0 to 60 (*) in 4 weeks](https://medium.com/@mslavescu/from-0-to-60-in-4-weeks-f6463ffe28a9)
    - [Why do we need Open Source Self Driving Car development and how to get started](https://medium.com/@mslavescu/why-do-we-need-open-source-self-driving-car-development-and-how-to-get-started-f71d36f2bae4)
  - [A few updates on AI and Self Driving Cars](https://chatbotslife.com/a-few-updates-on-ai-and-self-driving-cars-df48fdaa0733)
  - [Understand instead of Memorize](https://medium.com/@mslavescu/understand-instead-of-memorize-780790bd815)
  - [What is next in OSSDC.org?](https://becominghuman.ai/what-is-next-in-ossdc-org-3610f75794f3)
  - [What about putting a computer vision processor on the camera or sensor platform itself?](https://medium.com/@mslavescu/what-about-putting-a-computer-vision-processor-on-the-camera-or-sensor-platform-itself-d0622b24f5c)
  - [Learn by example](https://medium.com/@mslavescu/learn-by-example-f539ad814117)
  - [Get ready to Race AI with us at OSSDC.org](https://medium.com/@mslavescu/get-ready-to-race-ai-with-us-at-ossdc-org-b741e266e362)
  - [Live Visual Speed Recognition at 5000FPS - OSSDC PS3/PS4 Simulator running GT5 on PS3](https://medium.com/@mslavescu/live-visual-speed-recognition-at-5000fps-ossdc-ps3-ps4-simulator-running-gt5-on-ps3-85a435c0fd4e)
  
  
We use docker to easily run the OSSDC code, check these docker instructions:
- https://github.com/OSSDC/self-driving-car-1

For a consolidated view about OSSDC platform and related work, check the hacking book we are creating here:
- [OSSDC Hacking Book](https://github.com/OSSDC/OSSDC-Hacking-Book)
