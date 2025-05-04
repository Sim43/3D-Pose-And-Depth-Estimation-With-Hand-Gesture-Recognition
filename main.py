import cv2
import mediapipe as mp
import argparse
import os

# Function to detect and draw pose landmarks
def detect_and_draw_pose(image, visibility_threshold=0.5):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))
    connection_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))

    pose = mp_pose.Pose()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    annotated_image = image.copy()

    if results.pose_landmarks:
        # Use MediaPipe's built-in drawing for cleaner output
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=connection_spec
        )

        # Extract landmarks if needed
        landmark_dict = {}
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            name = mp_pose.PoseLandmark(i).name
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            landmark_dict[name] = (x, y, landmark.visibility)

        return annotated_image, landmark_dict

    return image, None

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Pose detection from image, video or camera.')
    parser.add_argument('--input', required=True, help='Path to image/video or camera index (e.g., 0)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Visibility threshold')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    elif os.path.isfile(args.input):
        ext = os.path.splitext(args.input)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            image = cv2.imread(args.input)
            annotated_image, landmarks = detect_and_draw_pose(image, args.threshold)

            if landmarks:
                for name, (x, y, vis) in landmarks.items():
                    print(f'{name}: ({x}, {y}), visibility: {vis:.2f}')

            cv2.imshow("Pose Detection", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        else:
            cap = cv2.VideoCapture(args.input)
    else:
        print("Invalid input source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, _ = detect_and_draw_pose(frame, args.threshold)
        cv2.imshow('Pose Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
