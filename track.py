import cv2
import numpy as np

class FaceEyeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.detector_params = cv2.SimpleBlobDetector_Params()
        self.detector_params.filterByArea = True
        self.detector_params.maxArea = 1500
        self.detector = cv2.SimpleBlobDetector_create(self.detector_params)

    def detect_faces(self, img):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        coords = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(coords) > 0:
            biggest = max(coords, key=lambda x: x[3])
            x, y, w, h = biggest
            frame = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            return frame
        return None

    def detect_eyes(self, img):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_frame, 1.3, 5)
        width = img.shape[1]
        height = img.shape[0]
        left_eye = None
        right_eye = None
        for x, y, w, h in eyes:
            if y > height / 1.5:
                continue
            eyecenter = x + w / 2
            if eyecenter < width * 0.5:
                left_eye = img[y:y + h, x:x + w]
            else:
                right_eye = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return left_eye, right_eye

    @staticmethod   
    def cut_eyebrows(img):
        height, width = img.shape[:2]
        eyebrow_h = int(height / 4)
        return img[eyebrow_h:height, 0:width]

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    detector = FaceEyeDetector()

    while True:
        _, frame = cap.read()
        face_frame = detector.detect_faces(frame)

        if face_frame is not None:
            eyes = detector.detect_eyes(face_frame)
            left_eye, right_eye = eyes

            if left_eye is not None and right_eye is not None:

                # Ensure the left and right eyes have the same height
                min_height = min(left_eye.shape[0], right_eye.shape[0])
                left_eye = left_eye[:min_height, :]
                right_eye = right_eye[:min_height, :]

                # Resize the eyes to have the same width
                width = max(left_eye.shape[1], right_eye.shape[1])
                left_eye = cv2.resize(left_eye, (width, min_height))
                right_eye = cv2.resize(right_eye, (width, min_height))

                # Display the two eyes side by side
                eyes_combined = np.hstack((left_eye, right_eye))
                cv2.imshow("image", eyes_combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
