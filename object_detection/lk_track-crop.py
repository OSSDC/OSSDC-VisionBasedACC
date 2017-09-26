#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track-crop.py -v [<video_source>]


Keys
----
ESC - exit
'''
from __future__ import print_function

'''
cap=None
sct=None
'''

# Python 2/3 compatibility

import cv2
from imutils.video import FPS
from video_tools import *

lk_params = dict( winSize  = (15, 15),#(15, 15),
                  maxLevel = 3,#2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0.01))

feature_params = dict( maxCorners = 5000, #500,
                       qualityLevel = 0.1, #0.3,
                       minDistance = 3, #7,
                       blockSize = 3 ) #7 )

#procWidth = 1280 #640   # processing width (x resolution) of frame
#procHeight = 720   # processing width (x resolution) of frame

track_len = 25
detect_interval = 15
tracks = []
frame_idx = 0

#def main():

print(__doc__)

print("procHeight",procHeight,"procWidth",procWidth)
print("track_len",track_len,"detect_interval",detect_interval,"tracks",tracks,"frame_idx",frame_idx)
print(lk_params)
print(feature_params)

#run()
'''
if __name__ == '__main__':
    main()
'''

#def run():
fps = FPS().start()
fpsValue=0
#frame = cap.read()
#if True:
'''
flag=True

frameCnt=0
prevFrameCnt=0
prevTime=getCurrentClock()
'''
while True:
    #print("frameCnt",frameCnt)
    if sct is not None or webcam or cap.grab():
        if sct is not None:
            frame = numpy.asarray(sct.grab(mon))
        else:
            if webcam:
                frame = cap.read()
            else:
                flag, frame = cap.retrieve()
        if not flag:
            print("frameCnt",frameCnt)
            continue
        else:
            frameCnt=frameCnt+1
            nowMicro = getCurrentClock()
            delta = (nowMicro-prevTime).total_seconds()
            #print("%f " % (delta))
            if delta>=1.0:
                fpsValue = ((frameCnt-prevFrameCnt)/delta) 
                print("FPS = %0.2f" % fpsValue)
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

            #frame = cv2.blur(frame,(5,5))
            frame = cv2.resize(frame,(procWidth,procHeight))
            #frame = cv2.blur(frame,(10,10))

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

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
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %5d FPS = %0.2f' % (len(tracks), fpsValue))

            if frame_idx % detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])


            frame_idx += 1
            prev_gray = frame_gray
            cv2.imshow('main', vis)
            fps.update()

            ch = cv2.waitKey(1)
            if ch == 27:
                break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
