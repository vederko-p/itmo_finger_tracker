
import cv2
from tracker import PositionTracker
from models.mediapipe_hands import MediapipeModel


POINT_COLOR = (0, 0, 255)

mediapipe_model = MediapipeModel()
position_tracker = PositionTracker(n_to_hold=20)


def main():
    cap = cv2.VideoCapture(0)
    while True:

        # get frame:
        ret, frame = cap.read()
        if not ret:
            break

        # detect finger print:
        cords = mediapipe_model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cords is not None:
            position_tracker.append(cords)

        # add points:
        for xy in position_tracker.cords:
            cv2.circle(frame, (xy[0], xy[1]), 10, POINT_COLOR, -1)

        # show frame:
        cv2.imshow('MyWindow', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
