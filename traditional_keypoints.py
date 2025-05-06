import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac
import time

class ImageMatch:
    def __init__(self, image):
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        self.gray = cv2.normalize(self.gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Hyperparameters
        self.s = 2
        self.num_octaves = 4
        self.contrast_cutoff = 0.03
        self.r_cutoff = 10
        self.M_eig_ratio_cutoff = 15
        self.num_orientation_bins = 36
        self.num_descriptor_parts = 4
        self.sift_window = 4 * self.num_descriptor_parts
        self.num_descriptor_bins = 8
        self.RANSAC_it = 5000
        self.RANSAC_delta = 5
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        
        # Store feature points and descriptors
        self.keypoints = None
        self.descriptors = None
    
    def get_feature_points(self):
        """Detect SIFT keypoints"""
        self.keypoints, self.descriptors = self.sift.detectAndCompute((self.gray * 255).astype(np.uint8), None)
    
    def plot_feature_points(self):
        """Plot keypoints on image"""
        img = cv2.drawKeypoints(self.image, self.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Feature Points')
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    
    @staticmethod
    def get_match_points(ref, cur):
        """Match features between reference and current image"""
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(ref.descriptors, cur.descriptors, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
        
        # Extract matched keypoints
        ref_pts = np.float32([ref.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        cur_pts = np.float32([cur.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        return ref_pts, cur_pts, good_matches
    
    @staticmethod
    def plot_match_points(ref, cur, ref_pts, cur_pts, matches):
        """Plot matched points between images"""
        # Create a composite image
        h1, w1 = ref.image.shape[:2]
        h2, w2 = cur.image.shape[:2]
        composite = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        composite[:h1, :w1] = ref.image
        composite[:h2, w1:w1+w2] = cur.image
        
        # Draw lines between matches
        for (x1, y1), (x2, y2) in zip(ref_pts, cur_pts):
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2 + w1), int(y2))
            cv2.line(composite, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(composite, pt1, 3, (0, 0, 255), -1)
            cv2.circle(composite, pt2, 3, (0, 0, 255), -1)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        plt.title('Matched Points')
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
    
    @staticmethod
    def get_pose(ref_pts, cur_pts):
        """Estimate homography using RANSAC"""
        # Robustly estimate homography with RANSAC
        model, inliers = ransac(
            (ref_pts, cur_pts),
            ProjectiveTransform, min_samples=4,
            residual_threshold=5, max_trials=5000
        )
        
        inlier_ref_pts = ref_pts[inliers]
        inlier_cur_pts = cur_pts[inliers]
        
        return model.params, inlier_ref_pts, inlier_cur_pts
    
    @staticmethod
    def plot_pose(ref, cur, h, ref_pts, cur_pts):
        """Plot the estimated homography"""
        # Create a composite image
        h1, w1 = ref.image.shape[:2]
        h2, w2 = cur.image.shape[:2]
        composite = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        composite[:h1, :w1] = ref.image
        composite[:h2, w1:w1+w2] = cur.image
        
        # Draw the reference frame
        box_ref = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0], [0, 0]]).reshape(-1, 1, 2)
        cv2.polylines(composite, [np.int32(box_ref)], True, (255, 0, 0), 2)
        
        # Transform the reference frame to current image
        box_cur = cv2.perspectiveTransform(box_ref, h)
        box_cur[:, :, 0] += w1  # Offset x-coordinates
        cv2.polylines(composite, [np.int32(box_cur)], True, (255, 0, 0), 2)
        
        # Draw lines between corresponding points
        for (x1, y1), (x2, y2) in zip(ref_pts, cur_pts):
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2 + w1), int(y2))
            cv2.line(composite, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(composite, pt1, 3, (0, 0, 255), -1)
            cv2.circle(composite, pt2, 3, (0, 0, 255), -1)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
        plt.title('Pose Estimation')
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Read first frame to set up reference
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
    
    # Select a region of interest (ROI) for reference
    x, y, w, h = 70, 40, 210, 210  # Similar to MATLAB example coordinates
    ref_img = frame[y:y+h, x:x+w]
    
    # Create reference image match object
    ref = ImageMatch(ref_img)
    ref.get_feature_points()
    print(f"Reference: Found {len(ref.keypoints)} keypoints")
    ref.plot_feature_points()
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create current image match object
        cur = ImageMatch(frame)
        cur.get_feature_points()
        
        # Match points
        ref_pts, cur_pts, matches = ImageMatch.get_match_points(ref, cur)
        
        if len(matches) >= 4:
            # Estimate pose
            h, inlier_ref_pts, inlier_cur_pts = ImageMatch.get_pose(ref_pts, cur_pts)
            
            # Draw matches and pose
            img_matches = cv2.drawMatches(
                ref.image, ref.keypoints,
                cur.image, cur.keypoints,
                matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            # Draw the reference frame transformed to current view
            h_total, w_total = ref.image.shape[:2]
            box_ref = np.float32([[0, 0], [0, h_total-1], [w_total-1, h_total-1], [w_total-1, 0]]).reshape(-1, 1, 2)
            box_cur = cv2.perspectiveTransform(box_ref, h)
            cv2.polylines(frame, [np.int32(box_cur)], True, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Matches', img_matches)
            cv2.imshow('Pose Estimation', frame)
        else:
            print(f"Not enough matches found: {len(matches)}")
            cv2.imshow('Current Frame', frame)
        
        # Exit on ESC
        if cv2.waitKey(1) == 27:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()