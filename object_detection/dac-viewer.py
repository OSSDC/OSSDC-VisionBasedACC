#import xml.etree.ElementTree as ET
import lxml.etree

import cv2
import glob
import os
import argparse

from common import *
from datetime import datetime

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

precision = 10
def getCurrentClock():
    #return time.clock()
    return datetime.now()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", help="path to the images")

args = vars(ap.parse_args())

url = args.get("images", None)

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
