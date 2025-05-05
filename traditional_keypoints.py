import cv2
import numpy as np

class TraditionalPoseEstimator:
    def __init__(self):
        # Initialize HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_person(self, frame):
        # Resize for faster processing
        resized = cv2.resize(frame, (640, 480))
        boxes, _ = self.hog.detectMultiScale(resized, winStride=(8, 8))
        return boxes, resized

    def estimate_pose(self, frame, box):
        x, y, w, h = box
        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pose_points = []

        if contours:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)

            # Approximate head as the topmost point
            top = tuple(hull[hull[:, :, 1].argmin()][0])
            bottom = tuple(hull[hull[:, :, 1].argmax()][0])
            center_x = int((top[0] + bottom[0]) / 2)

            # Fake shoulder, hip, knee positions based on height
            height = bottom[1] - top[1]
            pose_points = [
                (x + center_x, y + top[1]),                  # Head
                (x + center_x, y + int(top[1] + 0.25*height)), # Shoulder
                (x + center_x, y + int(top[1] + 0.5*height)),  # Hip
                (x + center_x, y + int(top[1] + 0.75*height))  # Knee
            ]
        return pose_points

    def draw_pose(self, frame, points):
        for pt in points:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        for i in range(len(points)-1):
            cv2.line(frame, points[i], points[i+1], (255, 0, 0), 2)

    def process(self, frame):
        boxes, processed = self.detect_person(frame)
        for box in boxes:
            pose_points = self.estimate_pose(processed, box)
            self.draw_pose(processed, pose_points)
        return processed

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # You can replace 0 with a video path

    estimator = TraditionalPoseEstimator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = estimator.process(frame)
        cv2.imshow("Traditional Pose Estimation", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
