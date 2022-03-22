from math import sqrt
from numpy import ndarray
from cv2 import cvtColor, COLOR_BGR2RGB
from typing import Union
from mediapipe import solutions


class Hand_Detector:

    def __init__(self, detection_confidence: float = 0.42, tracking_confidence: float = 0.6, max_hands: int = 2):
        self.detect = solutions.hands.Hands(static_image_mode=False, max_num_hands=max_hands,
                                               min_detection_confidence=detection_confidence,
                                               min_tracking_confidence=tracking_confidence).process

    def _detect_hands(self, img: ndarray) -> list[dict]:
        img_rgb = cvtColor(img, COLOR_BGR2RGB)
        results = self.detect(img_rgb)
        self.all_hands = []
        h, w, c = img.shape
        if results.multi_hand_landmarks:
            for hand_type, hand_lms in zip(results.multi_handedness, results.multi_hand_landmarks):
                hand = {}

                #lm_list
                lm_list = []
                x_list = []
                y_list = []
                for lm in hand_lms.landmark:
                    px, py, pz = int(lm.x * 100000), int(lm.y *
                                                         100000), int(lm.z * 100000)
                    lm_list.append([px, py, pz])
                    x_list.append(px)
                    y_list.append(py)

                ## box
                x_min, x_max = min(x_list), max(x_list)
                y_min, y_max = min(y_list), max(y_list)
                box_w = x_max - x_min
                box_h = y_max - y_min
                bbox = [x_min, y_min, box_w, box_h]
                cx = bbox[0] + (bbox[2] // 2)
                cy = bbox[1] + (bbox[3] // 2)

                hand["lm_list"] = lm_list
                hand["box"] = bbox
                hand["center"] = [cx, cy]

                if hand_type.classification[0].label == "Right":
                    hand["type"] = "left"
                else:
                    hand["type"] = "right"

                hand["fingers"] = self._finger_detection(hand)
                self.all_hands.append(hand)

        return self.all_hands

    def _finger_detection(self, hand: list[dict]) -> list[int]:

        lm_list = hand["lm_list"]
        fingers = []
        tip_ids = [4, 8, 12, 16, 20]
        direction = self._get_orientation(lm_list)

        #thumb
        if direction > 0:
            if lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0]:
                fingers.append(1)
            else:
                fingers.append(0)

        # all other fingers
        for id in range(1, 5):
            if lm_list[tip_ids[id]][1] < lm_list[tip_ids[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def _get_orientation(self, lm_list: list[list[int]]) -> int:

        points = [5, 9, 13, 17]
        direction = 0
        for i in range(0, 3):
            if lm_list[points[i]] > lm_list[points[i+1]]:
                direction += 1
            else:
                direction -= 1
        return direction

    def _get_distance(self, p1: list[int], p2: list[int]) -> float:
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def analyze_hands(self) -> tuple[bool, list[str]]:

        found = False
        rv = []
        #loop over hands
        for hand in self.all_hands:

            box = hand['box']
            lm_list = hand['lm_list']
            type = hand['type']
            fingers = hand['fingers']

            if self._gesture_spock(lm_list, fingers):
                rv.append('spock')
                found = True

        return rv if found else None

    def search_gestures(self, img: ndarray) -> Union[list[str],None]:
        self._detect_hands(img)
        if len(self.all_hands) > 0:
            return self.analyze_hands()
        else:
            return None

    def _gesture_spock(self, lm_list: list[list[int]], fingers: list[int]) -> bool:

        index_middle = self._get_distance(lm_list[8], lm_list[12])
        pinky_ring = self._get_distance(lm_list[16], lm_list[20])
        middle_ring = self._get_distance(lm_list[12], lm_list[16])

        top_distance = self._get_distance(lm_list[12], lm_list[16])
        mid_distance = self._get_distance(lm_list[11], lm_list[15])
        bot_distance = self._get_distance(lm_list[10], lm_list[14])

        if middle_ring > (index_middle+pinky_ring)*1.2 and all(fingers) and top_distance > mid_distance > bot_distance and top_distance > bot_distance * 1.4:
            return True
        else:
            return False
