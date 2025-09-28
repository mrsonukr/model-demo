# MediaPipe Pose Detection - Google Colab Version
# Run each section separately in Colab cells

# =============================================================================
# CELL 1: Install and Import Dependencies
# =============================================================================
!pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import numpy as np
from IPython.display import display, clear_output
from google.colab.output import eval_js
from base64 import b64decode
import time

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# For displaying images in Colab
from google.colab.patches import cv2_imshow

print("✅ Dependencies installed and imported!")

# =============================================================================
# CELL 2: Camera Capture Function for Colab
# =============================================================================
def capture_frame_from_camera():
    """Capture a single frame from webcam in Colab"""
    js = """
    async function capture() {
        const video = document.createElement('video');
        document.body.appendChild(video);
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            
            video.srcObject = stream;
            await video.play();
            
            // Wait for video to stabilize
            await new Promise(resolve => setTimeout(resolve, 500));
            
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            // Clean up
            stream.getTracks().forEach(track => track.stop());
            video.remove();
            
            return canvas.toDataURL('image/jpeg', 0.8);
        } catch (error) {
            console.error('Camera error:', error);
            return null;
        }
    }
    capture();
    """
    
    try:
        data = eval_js(js)
        if data:
            binary = b64decode(data.split(',')[1])
            nparr = np.frombuffer(binary, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
    except Exception as e:
        print(f"Error capturing frame: {e}")
    return None

# =============================================================================
# CELL 3: Angle Calculation Function
# =============================================================================
def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point (vertex)
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# =============================================================================
# CELL 4: Basic Pose Detection (Single Frame)
# =============================================================================
def detect_pose_single_frame():
    """Detect pose on a single frame"""
    print("📷 Capturing frame for pose detection...")
    frame = capture_frame_from_camera()
    
    if frame is None:
        print("❌ Failed to capture frame")
        return
    
    # Setup MediaPipe
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            print(f"✅ Detected {len(landmarks)} pose landmarks")
            
            # Show specific landmark info
            try:
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                print(f"Left Shoulder - x: {left_shoulder.x:.3f}, y: {left_shoulder.y:.3f}, visibility: {left_shoulder.visibility:.3f}")
            except:
                print("Could not extract shoulder landmark")
        else:
            print("❌ No pose detected")
    
    # Display result
    print("🖼️ Displaying result:")
    cv2_imshow(image)
    return image, results

# Run single frame detection
# detect_pose_single_frame()

# =============================================================================
# CELL 5: Angle Detection with Visualization
# =============================================================================
def detect_pose_with_angle():
    """Detect pose and calculate arm angle"""
    print("📷 Capturing frame for angle detection...")
    frame = capture_frame_from_camera()
    
    if frame is None:
        print("❌ Failed to capture frame")
        return
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Process image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks and calculate angle
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            try:
                # Get coordinates for left arm
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Visualize angle
                cv2.putText(image, f'{angle:.1f}°', 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                print(f"💪 Left arm angle: {angle:.1f}°")
                
                # Also calculate right arm angle
                try:
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    
                    cv2.putText(image, f'{r_angle:.1f}°', 
                               tuple(np.multiply(r_elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    print(f"💪 Right arm angle: {r_angle:.1f}°")
                except:
                    print("Could not calculate right arm angle")
                    
            except Exception as e:
                print(f"❌ Could not calculate angles: {e}")
            
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        else:
            print("❌ No pose detected")
    
    # Display result
    print("🖼️ Displaying result with angles:")
    cv2_imshow(image)
    return image

# Run angle detection
# detect_pose_with_angle()

# =============================================================================
# CELL 6: Bicep Curl Counter (Multi-frame simulation)
# =============================================================================
def bicep_curl_counter_demo(num_frames=10):
    """Simulate bicep curl counting over multiple frames"""
    print(f"🏋️ Starting bicep curl counter demo ({num_frames} frames)")
    print("💡 Move your left arm up and down to simulate bicep curls!")
    
    # Counter variables
    counter = 0
    stage = None
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for i in range(num_frames):
            print(f"\n📷 Frame {i+1}/{num_frames} - Get ready...")
            time.sleep(2)  # Give time to pose
            
            frame = capture_frame_from_camera()
            if frame is None:
                print("❌ Failed to capture frame, skipping...")
                continue
            
            # Process image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                try:
                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    # Calculate angle
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                    # Bicep curl logic
                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage == 'down':
                        stage = "up"
                        counter += 1
                        print(f"🎉 CURL COMPLETED! Count: {counter}")
                    
                    # Visualize angle
                    cv2.putText(image, f'{angle:.1f}°', 
                               tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Create status box
                    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
                    
                    # Rep counter
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    # Stage indicator
                    cv2.putText(image, 'STAGE', (65,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage if stage else 'READY', (60,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    print(f"📊 Angle: {angle:.1f}°, Stage: {stage}, Reps: {counter}")
                    
                except Exception as e:
                    print(f"❌ Error processing landmarks: {e}")
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
            else:
                print("❌ No pose detected in this frame")
            
            # Display result
            clear_output(wait=True)
            cv2_imshow(image)
            
    print(f"\n🎯 Final Results: {counter} bicep curls completed!")
    return counter

# =============================================================================
# CELL 7: Interactive Demo Functions
# =============================================================================
def show_all_pose_landmarks():
    """Show all available pose landmarks"""
    print("📍 All MediaPipe Pose Landmarks:")
    for i, landmark in enumerate(mp_pose.PoseLandmark):
        print(f"{i:2d}. {landmark.name}")

def run_pose_analysis():
    """Complete pose analysis with all features"""
    print("🤖 Running complete pose analysis...")
    
    frame = capture_frame_from_camera()
    if frame is None:
        return
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calculate multiple angles
            angles = {}
            try:
                # Left arm angle
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angles['Left Arm'] = calculate_angle(l_shoulder, l_elbow, l_wrist)
                
                # Right arm angle
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                angles['Right Arm'] = calculate_angle(r_shoulder, r_elbow, r_wrist)
                
                # Left knee angle
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angles['Left Knee'] = calculate_angle(l_hip, l_knee, l_ankle)
                
                print("📐 Joint Angles:")
                for joint, angle in angles.items():
                    print(f"  {joint}: {angle:.1f}°")
                
            except Exception as e:
                print(f"Could not calculate all angles: {e}")
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        cv2_imshow(image)

# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
print("""
🎯 GOOGLE COLAB MEDIAPIPE POSE DETECTION READY!

📋 Available Functions:
1. detect_pose_single_frame() - Basic pose detection
2. detect_pose_with_angle() - Pose detection with arm angles
3. bicep_curl_counter_demo(10) - Bicep curl counter (10 frames)
4. show_all_pose_landmarks() - List all available landmarks
5. run_pose_analysis() - Complete pose analysis

💡 Usage Examples:
   detect_pose_single_frame()
   detect_pose_with_angle() 
   bicep_curl_counter_demo(5)
   
⚠️  Important Notes:
- Grant camera permission when prompted
- Make sure you're visible in good lighting
- Keep some distance from camera for full body detection
- Functions work one frame at a time (no real-time video)

🚀 Ready to start! Run any function above to begin pose detection.
""")