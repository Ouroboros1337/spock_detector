# spock_detector
A wrapper for the mediapipe hand detection to find the spock hand.
You can add other gestures yourself see https://google.github.io/mediapipe/solutions/hands.html for lm_list point refernce.

Usage:

from spock_detector import Hand_Detector()

hand_detector = Hand_Detector()

# some way to get images (cv.readim, my rtsp library, other frame sources)

gestures = hand_detector.search_gestures(frame)

if gestures:
   if 'spock' in gestures:
       print('found spock')
