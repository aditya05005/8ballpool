import cv2
import numpy as np
from ultralytics import YOLO
from pocket_logic import PocketDetector

# --- CONFIGURATION ---
VIDEO_PATH = '/home/aditya/Documents/8ballpool/test2.mp4'
MODEL_PATH = '/home/aditya/Documents/8ballpool/yolov8_20epoc_imgsz832.onnx'
SKIP_FRAMES = 2 

# UPDATE THIS MAP based on your findings!
# We will use this to FILTER OUT everything except balls.
CLASS_NAMES = {
    0: "8-Ball",
    3: "Cue Ball",
    4: "Pocket_Ignore", # We will filter this
    6: "Solid",
    7: "Stripe",
    5: "railing"
}

# --- PHASE 1: UI FOR POCKET SELECTION ---
def select_pockets_ui(video_path):
    print("--- PHASE 1: POCKET SELECTION ---")
    print("Please click on the center of all 6 pockets.")
    print("Press 'r' to reset if you mess up.")
    print("Press 'c' or 'ENTER' when done.")
    
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        raise ValueError(f"Could not read video: {video_path}")

    selected_points = []
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((x, y))
            print(f"Pocket {len(selected_points)} selected at {x}, {y}")

    cv2.namedWindow("Select Pockets")
    cv2.setMouseCallback("Select Pockets", mouse_callback)

    while True:
        display_img = frame.copy()
        
        # Draw all selected points so far
        for i, p in enumerate(selected_points):
            cv2.circle(display_img, p, 15, (0, 255, 0), -1) # Green dot
            cv2.circle(display_img, p, 30, (0, 255, 0), 2)  # Ring
            cv2.putText(display_img, str(i+1), (p[0]+10, p[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Draw Instructions
        cv2.putText(display_img, f"Selected: {len(selected_points)}/6", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_img, "Click pockets. Press 'c' to confirm.", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Select Pockets", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == 13: # 'c' or Enter
            if len(selected_points) < 6:
                print("Warning: You selected fewer than 6 pockets. Press 'c' again to confirm anyway.")
                if cv2.waitKey(0) & 0xFF in [ord('c'), 13]:
                    break
            else:
                break
        elif key == ord('r'): # Reset
            selected_points = []
            print("Resetting points...")
        elif key == ord('q'): # Quit entirely
            exit()

    cv2.destroyWindow("Select Pockets")
    return selected_points

# --- RUN PHASE 1 ---
# This blocks the code until you finish clicking
pockets = select_pockets_ui(VIDEO_PATH)
print(f"Starting Phase 2 with pockets: {pockets}")

# --- PHASE 2: TRACKING LOOP ---

cap = cv2.VideoCapture(VIDEO_PATH)
# Radius 25 is usually a good "Kill Zone" size
detector = PocketDetector(pocket_coords=pockets, pocket_radius=25, cooldown_frames=15)
model = YOLO(MODEL_PATH, task='detect')

frame_count = 0
last_detections = np.empty((0, 2))
last_classes = np.empty((0))

print("--- PHASE 2: MATCH TRACKING STARTED ---")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    if frame_count % SKIP_FRAMES == 0:
        results = model(frame, verbose=False)
        
        if results[0].boxes:
            boxes = results[0].boxes.xywh.cpu().numpy()
            cls = results[0].boxes.cls.cpu().numpy()
            
            # --- CRITICAL: IGNORE POCKETS/RAILS FROM MODEL ---
            # We ONLY want the logic to see balls (IDs 0, 3, 6, 7)
            # This 'valid_mask' ignores Class 4 (Pocket) and Class 8 (Rail - if you have it)
            valid_mask = np.isin(cls, [0, 3, 6, 7]) 
            
            last_detections = boxes[valid_mask][:, :2]
            last_classes = cls[valid_mask]
            
        else:
            last_detections = np.empty((0, 2))
            last_classes = np.empty((0))

        # Update Logic
        total_score, events = detector.update(last_detections, last_classes)
        
        if events:
            for pocket_idx, ball_class_id in events:
                ball_name = CLASS_NAMES.get(ball_class_id, "Unknown Ball")
                print(f"EVENT: {ball_name} pocketed!")
                # Visual Flash
                cv2.putText(frame, f"{ball_name} POCKETED!", (300, 300), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # --- DRAWING ---
    # Draw the user-defined pockets
    for i, p in enumerate(pockets):
        cv2.circle(frame, p, 25, (0, 255, 0), 2)

    # Draw Balls
    for i, ball in enumerate(last_detections):
        x, y = int(ball[0]), int(ball[1])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        if len(last_classes) > i:
             c_id = int(last_classes[i])
             label = CLASS_NAMES.get(c_id, str(c_id))
             cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Draw Score
    cv2.putText(frame, f"Count: {detector.total_pocketed_count}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("8 Ball Match Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
