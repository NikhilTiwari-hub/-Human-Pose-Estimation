# -Human-Pose-Estimation
"Real-time Human Pose Estimation using MediaPipe and OpenCV. This project demonstrates detecting and visualizing human body landmarks efficiently."
import cv2
import mediapipe as mp

# Initialize Mediapipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    # Initialize video capture (webcam)
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video path
    
    # Setup Mediapipe Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Unable to access the webcam. Exiting...")
                break
            
            # Convert frame to RGB for Mediapipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process frame for pose detection
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                )
            
            # Display the frame with pose estimation
            cv2.imshow("Human Pose Estimation", frame)
            
            # Exit the loop when 'q' key is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    # Release video capture and close display windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
