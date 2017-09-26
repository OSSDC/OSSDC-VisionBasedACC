from imutils.video import WebcamVideoStream

import mss
import numpy as np
import time
import subprocess
import argparse
import cv2
import video
from common import anorm2, draw_str
from time import clock
from imutils.video import FPS
from imutils.video import WebcamVideoStream
import pafy

from datetime import datetime

#procWidth = 1920 #640   # processing width (x resolution) of frame
#procHeight = 1080   # processing width (x resolution) of frame
procWidth = 1280   # processing width (x resolution) of frame
procHeight = 720   # processing width (x resolution) of frame

precision = 10
def getCurrentClock():
    #return time.clock()
    return datetime.now()


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		if abs(refPt[0][0]-refPt[1][0]) > 10:
		    cropping = True
		else:
			cropping = False
		#cropping = False
		# draw a rectangle around the region of interest
		#cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
		#cv2.imshow("ssd", img)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-s", "--skipNr", help="skip frames nr")

args = vars(ap.parse_args())


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

url = args.get("video", None)

if url is None:
    url = "https://www.youtube.com/watch?v=uuQlMCMT71I"

skipNr=args.get("skipNr", None)

if skipNr is not None :
    skipNr = int(skipNr)
else:
    skipNr=0

print("skipNr", skipNr)

def getVideoURL(url):
    videoUrl = url
    video = pafy.new(url)
    streams = video.streams
    videoUrlList={}
    for s in streams:
        videoUrlList[s.resolution] = s.url
        #print(s.resolution, s.extension, s.get_filesize(), s.url)

    if videoUrlList.get("1280x720",None) is not None:
        videoUrl = videoUrlList.get("1280x720",None)
        print("1280x720")

    if videoUrlList.get("1920x1080",None) is not None:
        videoUrl = videoUrlList.get("1920x1080",None)
        print("1920x1080")
    return videoUrl

if "youtube." in url: 
    videoUrl = getVideoURL(url)
else:
    videoUrl = url

print("videoUrl=",videoUrl)

# 800x600 windowed mode
#mon = {'top': 100, 'left': 2020, 'width': 1280, 'height': 720}
#mon = {'top': 0, 'left': 1920, 'width': 1280, 'height': 720}
mon = {'top': 0, 'left': 0, 'width': 1280, 'height': 720}
sct = None
if "screen" in url:
    sct = mss.mss()

webcam=False
#webcam=True
cap = None

if sct is None:
    if webcam:
        #cap = WebcamVideoStream(src=""+str(videoUrl)+"").start()
        cap = WebcamVideoStream(videoUrl).start()
    else:
        cap = cv2.VideoCapture(videoUrl)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, procWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, procHeight)

#cap = cv2.VideoCapture(videoUrl)

count=50
#skip=2000
skip=skipNr

SKIP_EVERY=150 #pick a frame every 5 seconds

count=1000000
#skip=0 #int(7622-5)
SKIP_EVERY=0

every=SKIP_EVERY
initial_time = getCurrentClock()
flag=True

frameCnt=0
prevFrameCnt=0
prevTime=getCurrentClock()

showImage=False
showImage=True

processImage=False
processImage=True

zoomImage=0
#zoomImage=True
rclasses = []
rscores = []
rbboxes = []

record = False
#record = True

out = None
if record:
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    out = cv2.VideoWriter('output-'+timestr+'.mp4',fourcc, 30.0, (int(procWidth),int(procHeight)))

#output_side_length = int(1920/zoomImage)

#height_offset = int((height - output_side_length) / 2)
#width_offset = int((width - output_side_length) / 2)

flag = True

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = [(0, 0),([procWidth],procHeight)]
cropping = False

cv2.namedWindow("main")
cv2.setMouseCallback("main", click_and_crop)
