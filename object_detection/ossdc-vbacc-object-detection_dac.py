
# coding: utf-8

# In[1]:

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import glob
import os
import argparse
import lxml.etree

from datetime import datetime

slim = tf.contrib.slim

# In[2]:

#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#from skimage import io
import time
import subprocess

import pafy

from common import *

def parseXML(xmlfile):
 
    # create element tree object
    tree = lxml.etree.parse(xmlfile)
 
    # get root element
    entry = tree.getroot()
 
    filename = entry.xpath('/annotation/filename/text()')[0] 
    name = entry.xpath('/annotation/object/name/text()')[0] 
    xmin = entry.xpath('/annotation/object/bndbox/xmin/text()')[0] 
    #print("xmin",xmin)
    ymin = entry.xpath('/annotation/object/bndbox/ymin/text()')[0] 
    xmax = entry.xpath('/annotation/object/bndbox/xmax/text()')[0] 
    ymax = entry.xpath('/annotation/object/bndbox/ymax/text()')[0] 

    # create empty list for news items
    box = [filename,name,int(xmin),int(ymin),int(xmax),int(ymax)]
 
    return box

# In[3]:

import sys
sys.path.append('../')

precision = 10
def getCurrentClock():
    #return time.clock()
    return datetime.now()


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping, tracks

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
		    tracks = [] #reset tracking
		else:
			cropping = False
		#cropping = False
		# draw a rectangle around the region of interest
		#cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
		#cv2.imshow("ssd", img)

# In[4]:


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-s", "--skipNr", help="skip frames nr")
ap.add_argument("-m", "--modelType", help="model type")
ap.add_argument("-t", "--trackCnt", help="track max corners")
ap.add_argument("-w", "--webcam", help="webcam mode")
ap.add_argument("-r", "--resolution", help="resolution default (640,480)")
ap.add_argument("-f", "--framerate", help="frames per second, default 30")
#ap.add_argument("-i", "--images", help="path to the images")

args = vars(ap.parse_args())

#url = args.get("images", None)

modelType=args.get("modelType", None)

if modelType is None :
    modelType =  "ssd"

webcam=args.get("webcam", None)

if webcam is None or int(webcam)==0:
    webcam =  False
else:
    webcam = True

tracking=args.get("trackCnt", None)

if tracking is None:
    tracking = 0
else:
    tracking = int(tracking) 

framerate=args.get("framerate", None)

if framerate is None:
    framerate = 30
else:
    framerate = int(framerate) 


#procWidth = 1920 #640   # processing width (x resolution) of frame
#procHeight = 1080   # processing width (x resolution) of frame
procWidth = 1280   # processing width (x resolution) of frame
procHeight = int(procWidth*(1080/1920))   # processing width (x resolution) of frame

resolution=args.get("resolution", None)

if resolution is None:
    (procWidth,procHeight) = (640,480)
else:
    (procWidth,procHeight) = resolution.split(",") 

procWidth = int(procWidth)
procHeight = int(procHeight)

shapeWidth=512
shapeHeight=512
shapeWidth=300
shapeHeight=300

if modelType=="ssd":
    if shapeWidth==300:
        from nets import ssd_vgg_300, ssd_common, np_methods
    else:
        from nets import ssd_vgg_512, ssd_common, np_methods
    from preprocessing import ssd_vgg_preprocessing
    import visualization
elif modelType=="tensorflow":
    from utils import label_map_util
    from utils import visualization_utils as vis_util
    from collections import defaultdict
    from PIL import Image

print("procWidth",procWidth,"procHeight", procHeight)

#print("Test")

# In[5]:

# In[6]:


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

#A smooth drive in The Crew on PS4 - OSSDC Simulator ACC Train 30fps
#videoUrl = subprocess.Popen("youtube-dl.exe -f22 -g https://www.youtube.com/watch?v=uuQlMCMT71I", shell=True, stdout=subprocess.PIPE).stdout.read()
#videoUrl = videoUrl.decode("utf-8").rstrip()


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

origVideoUrl = url

if "youtube." in url: 
    videoUrl = getVideoURL(url)
else:
    videoUrl = url

# if the video argument is None, then we are reading from webcam
#videoUrl = args.get("video", None)

print("videoUrl=",videoUrl)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



# ## Post-processing pipeline
# 
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
# 
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.

# In[7]:

# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(shapeWidth, shapeHeight)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes



if modelType=="ssd":

    with tf.device('/gpu:0'):

        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        #gpu_options = tf.GPUOptions(allow_growth=True)
        #config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        #isess = tf.InteractiveSession(config=config)
        isess = tf.InteractiveSession() #config=tf.ConfigProto(log_device_placement=True))

        # ## SSD 300 Model
        # 
        # The SSD 300 network takes 300x300 image inputs. In order to feed any image, the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that even though it may change the ratio width / height, the SSD model performs well on resized images (and it is the default behaviour in the original Caffe implementation).
        # 
        # SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors.


        # Input placeholder.
        #net_shape = (300, 300)
        net_shape = (shapeWidth, shapeHeight)

        data_format = 'NHWC' #'NHWC' #'NCHW'
        img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        image_4d = tf.expand_dims(image_pre, 0)


        # Define the SSD model.
        reuse = True if 'ssd_net' in locals() else None
        if shapeWidth==300:
            ssd_net = ssd_vgg_300.SSDNet()
        else:
            ssd_net = ssd_vgg_512.SSDNet()

        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

        # Restore SSD model.
        if shapeWidth==300:
            ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'
        else:
            ckpt_filename = 'checkpoints/VGG_VOC0712_SSD_512x512_ft_iter_120000.ckpt'
        # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(isess, ckpt_filename)

        # SSD default anchor boxes.
        ssd_anchors = ssd_net.anchors(net_shape)



# In[10]:


if modelType=="tensorflow":

    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    #with tf.device('/gpu:0'):

    sess = tf.InteractiveSession(graph=detection_graph) #,config=tf.ConfigProto(log_device_placement=True)) #tf.InteractiveSession()

start_time = getCurrentClock()

from webcamvideostream import *

import mss
import numpy

'''
fpsValue=0

frameCnt=0
prevFrameCnt=0
prevTime=getCurrentClock()

for imgFile in sorted(glob.glob(url+"/*.jpg")):
    bi = parseXML(url+os.path.splitext(os.path.basename(imgFile))[0]+".xml")
    print(bi)
    key = cv2.waitKey(1)
    if  key == 27:
      break
    img = cv2.imread(imgFile)
    cv2.rectangle(img, (bi[2],bi[3]), (bi[4],bi[5]), (0, 255, 0), 2)
    frameCnt=frameCnt+1
    nowMicro = getCurrentClock()
    delta = (nowMicro-prevTime).total_seconds()
    #print("%f " % (delta))
    if delta>=1.0:
        fpsValue = ((frameCnt-prevFrameCnt)/delta)
        prevTime = getCurrentClock()
        prevFrameCnt=frameCnt
    
    draw_str(img, (20, 20), "FPS = %03.2f, Frame = %05d, Object = %8s, File = %10s" % (fpsValue,frameCnt,bi[1],bi[0]))

    cv2.imshow("tracking", img)
'''

# 800x600 windowed mode
#mon = {'top': 100, 'left': 2020, 'width': 1280, 'height': 720}
#mon = {'top': 0, 'left': 1920, 'width': 1280, 'height': 720}
mon = {'top': 0, 'left': 0, 'width': 1280, 'height': 720}
sct = None

def getCap(videoUrl):
    global sct
    if "screen" in url:
        sct = mss.mss()
    cap = None
    if sct is None:
        if webcam:
            #cap = WebcamVideoStream(src=""+str(videoUrl)+"").start()
            cap = WebcamVideoStream(videoUrl,(procWidth,procHeight),framerate)
            cap.start()
        else:
            cap = cv2.VideoCapture(videoUrl)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, procWidth)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, procHeight)
            cap.set(cv2.CAP_PROP_FPS, framerate)	
    return cap


cap = getCap(videoUrl)

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

cv2.namedWindow("ossdc.org source: " + origVideoUrl)
cv2.setMouseCallback("ossdc.org source: " + origVideoUrl, click_and_crop)

fpsValue=0

tracks = []

if tracking>0:
    lk_params = dict( winSize  = (15, 15),#(15, 15),
                    maxLevel = 3,#2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0.01))

    feature_params = dict( maxCorners = tracking, #5000 #500,
                        qualityLevel = 0.1, #0.3,
                        minDistance = 3, #7,
                        blockSize = 3 ) #7 )

    #procWidth = 1280 #640   # processing width (x resolution) of frame
    #procHeight = 720   # processing width (x resolution) of frame

    track_len = 25
    detect_interval = 15
    frame_idx = 0

prevZoomImage = zoomImage

imgFiles = sorted(glob.glob(url+"/*.jpg"))
imgCnt=0
flag=True

if 1==1:
    while True:
        #frame = cap.read()
        #if True:
        if imgFiles[imgCnt] is not None or sct is not None or webcam or cap.grab():
            if sct is not None:
                frame = numpy.asarray(sct.grab(mon))
            else:
                if webcam:
                    frame = cap.read()
                elif imgFiles[imgCnt] is not None:
                    imgFile = imgFiles[imgCnt]
                    print("imgFile",imgCnt,"=",imgFile)
                    frame = cv2.imread(imgFile)
                    frame = cv2.resize(frame,(int(640*2),int(360*2)))
                    imgCnt=imgCnt+1
                else:
                    flag, frame = cap.retrieve()    
            if not flag:
                    continue
            else:
                frameCnt=frameCnt+1
                nowMicro = getCurrentClock()
                delta = (nowMicro-prevTime).total_seconds()
                #print("%f " % (delta))
                if delta>=1.0:
                    fpsValue = ((frameCnt-prevFrameCnt)/delta) 
                    print("FPS = %3.2f, Track points = %5d, Frame = %6d" % (fpsValue,len(tracks), frameCnt))
                    prevTime = nowMicro
                    prevFrameCnt=frameCnt

                if skip>0:
                    skip=skip-1
                    continue
                
                if every>0:
                    every=every-1
                    continue
                every=SKIP_EVERY
                
                count=count-1
                if count==0:
                    break

                img = frame
                if processImage:    
                    if zoomImage>0:
                        #crop center of image, crop width is output_side_length
                        output_side_length = int(procWidth/zoomImage)
                        height, width, depth = frame.shape
                        #print (height, width, depth)
                        height_offset = int((height - output_side_length) / 2)
                        width_offset = int((width - output_side_length) / 2)
                        #print (height, width, depth, height_offset,width_offset,output_side_length)

                        #crop based on zoomImage value
                        img = frame[height_offset:height_offset + output_side_length,width_offset:width_offset + output_side_length]

                        
                    if zoomImage!=prevZoomImage:
                        #reset tracking
                        tracks = []
                        prevZoomImage = zoomImage

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    start_time = getCurrentClock()

                    if cropping:
                        fullImg = img.copy()
                        img=img[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]

                    if modelType=="ssd":
                        rclasses, rscores, rbboxes =  process_image(img)
                    elif modelType=="tensorflow":
                        # the array based representation of the image will be used later in order to prepare the
                        # result image with boxes and labels on it.
                        cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        pil_im = Image.fromarray(cv2_im)
                        image_np = load_image_into_numpy_array(pil_im)
                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)

                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        # Each box represents a part of the image where a particular object was detected.
                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        # Each score represent how level of confidence for each of the objects.
                        # Score is shown on the result image, together with the class label.
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                        # Actual detection.
                        (boxes, scores, classes, num) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                            
                    if len(rclasses)>0:
                        nowMicro = getCurrentClock()
                        if modelType=="ssd":
                            print("# %s - %s - %0.4f seconds ---" % (frameCnt,rclasses.astype('|S3'), (nowMicro - start_time).total_seconds()))
                        elif modelType=="tensorflow":
                            print("# %s - %s - %0.4f seconds ---" % (frameCnt, classes.astype('|S3'), (nowMicro - start_time).total_seconds()))
                        start_time = nowMicro
                    if showImage:
                        if modelType=="ssd":
                            visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
                        elif modelType=="tensorflow":
                            # Visualization of the results of a detection.
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                img,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8)

                            #img = image_np
                        if cropping:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            cv2.imshow("crop",img)

                    if tracking>0:
                        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        if len(tracks) > 0:
                            img0, img1 = prev_gray, frame_gray
                            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                            d = abs(p0-p0r).reshape(-1, 2).max(-1)
                            good = d < 1
                            new_tracks = []
                            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                                if not good_flag:
                                    continue
                                tr.append((x, y))
                                if len(tr) > track_len:
                                    del tr[0]
                                new_tracks.append(tr)
                                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                            tracks = new_tracks
                            if(showImage):
                                cv2.polylines(img, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                                #draw_str(img, (20, 20), 'track count: %5d FPS = %0.2f' % (len(tracks), fpsValue))
                        frame_idx += 1
                        prev_gray = frame_gray

                        if frame_idx % detect_interval == 0:
                            mask = np.zeros_like(frame_gray)
                            mask[:] = 255
                            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                                cv2.circle(mask, (x, y), 5, 0, -1)
                            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                            if p is not None:
                                for x, y in np.float32(p).reshape(-1, 2):
                                    tracks.append([(x, y)])

                    if cropping:
                        img = fullImg

                    #if record == 9:
                    #     cv2.imwrite('frame'.png',img)

                if showImage:
                    #visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
                    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
                    if processImage:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    if cropping:
                        try:
                            #if len(tracks)<5:
                            #    print('\a')
                            draw_str(img, (20, 20), "FPS = %3.2f, Track points = %5d, Frame = %6d" % (fpsValue,len(tracks), frameCnt))
                            cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
                            cv2.imshow("ossdc.org source: " + origVideoUrl, img)
                        except Exception:
                            pass
                    else:
                        #draw_str(img, (20, 20), 'track count: %5d FPS = %0.2f' % (len(tracks), fpsValue))
                        #if len(tracks)<5:
                        #    print('\a')
                        draw_str(img, (20, 20), "FPS = %3.2f, Track points = %5d, Frame = %6d" % (fpsValue,len(tracks), frameCnt))
                        cv2.imshow("ossdc.org source: " + origVideoUrl, img)
                if record:
                    #if processImage:
                        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    newimage = cv2.resize(img,(procWidth,procHeight))
                    out.write(newimage)
        key = cv2.waitKey(1)
        if  key == 27:
            break
        elif key == ord('c'):
            url = input("Enter new url: ")
            cv2.destroyWindow("ossdc.org source: " + origVideoUrl)
            origVideoUrl = url
            cv2.namedWindow("ossdc.org source: " + origVideoUrl)
            cv2.setMouseCallback("ossdc.org source: " + origVideoUrl, click_and_crop)

            videoUrl = getVideoURL(url)
            cap = getCap(videoUrl)
        elif key == ord('u'):
            showImage= not(showImage)
        elif key == ord('p'):
            processImage= not(processImage)
        elif key == ord('z'):
            zoomImage=zoomImage+1
            if zoomImage==10:
                zoomImage=0
        elif key == ord('x'):
            zoomImage=zoomImage-1
            if zoomImage<0:
                zoomImage=0

nowMicro = getCurrentClock()
print("# %s -- %0.4f seconds - FPS: %0.4f ---" % (frameCnt, (nowMicro - initial_time).total_seconds(), frameCnt/(nowMicro - initial_time).total_seconds()))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



