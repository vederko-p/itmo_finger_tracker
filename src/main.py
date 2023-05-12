
import cv2


def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        # find key_points coords
        # add coords into tracker
        # draw coords on frame

        if not ret:
            break
        cv2.imshow('MyWindow', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

