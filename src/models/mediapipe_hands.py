
import mediapipe as mp
import numpy as np


class MediapipeModel:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=True,
            max_num_hands=1
        )

    def __call__(self, img):
        h, w, _ = img.shape
        results = self.hands.process(img)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                finger_tip = hand_landmarks.landmark[8]
                return int(finger_tip.x * w), int(finger_tip.y * h)


class MediapipeHandDetector:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=True,
            max_num_hands=1
        )
    
    def make_square(self, box):
        const = 3.078343710504076
        box = box
        cf = int(np.linalg.norm(np.array(box[2:])-np.array(box[:2])) / const)
        x_dist = box[3] - box[1]
        y_dist = box[2] - box[0]
        if x_dist >  y_dist:
            razn = x_dist - y_dist
            box[2] += razn
        elif y_dist >  x_dist:
            razn = y_dist - x_dist
            box[3] += razn
        box[0], box[1], box[2], box[3] =  box[0] - cf, box[1] - cf, box[2] + cf, box[3] + cf
        return box

    def convert_to_bbox(self, box):
        box = np.array(box)
        x_min = np.min(box[:,0]) - 40
        x_max = np.max(box[:,0]) + 40
        y_min = np.min(box[:,1]) - 40
        y_max = np.max(box[:,1]) + 40
        return [x_min,y_min,x_max,y_max]

    def __call__(self, img):
        h, w, _ = img.shape
        results = self.hands.process(img)
        prebox = []
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                hand = []
                for finger_tip in hand_landmarks.landmark:
                    hand.append([int(finger_tip.x * w), int(finger_tip.y * h)])
                prebox.append(self.make_square(self.convert_to_bbox(hand)))
        return prebox