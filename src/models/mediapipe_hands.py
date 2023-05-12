
import mediapipe as mp


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
