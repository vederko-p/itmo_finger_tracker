import numpy as np
from PIL import Image
import cv2
from src.tracker import PositionTracker
from src.models.mediapipe_hands import MediapipeModel, MediapipeHandDetector
from src.models.graphormer import HandTopDownRecognition


POINT_COLOR = (0, 0, 255)

# mediapipe_model = MediapipeModel()
detector = MediapipeHandDetector()
hand = HandTopDownRecognition()
position_tracker = PositionTracker(n_to_hold=20)

def main(path):
    cap = cv2.VideoCapture(path)
    frame_width,frame_height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(f"output/output.mp4",0x7634706d, 24.0, (frame_width,frame_height), True)

    while (cap.isOpened()):
        # get frame:
        ret, frame = cap.read()
        if not ret:
            break

        # detect hand:
        boxes = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # add points:
        if len(boxes) > 0:
            for points in boxes:
                img = Image.fromarray(frame)
                img_size = img.size
                crop = img.crop(points)
                keypoints = hand.get_keypoints(crop)
                keypoints = (keypoints + 1) * 0.5
                keypoints[:,:,0] = (keypoints[:,:,0] * crop.size[0]) + points[0]
                keypoints[:,:,1] = (keypoints[:,:,1] * crop.size[1]) + points[1]
                keypoints = keypoints.cpu().detach().numpy().astype(int)
                position_tracker.append(keypoints[0][8])
                for xy in position_tracker.cords:
                    cv2.circle(frame, xy, 5, POINT_COLOR, -1)
                    
        out.write(np.uint8(frame))
    out.release()



if __name__ == '__main__':
    main('/root/other/itmo_finger_tracker/videos/test.mp4')
