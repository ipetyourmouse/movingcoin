import cv2
import time
import numpy as np
import mediapipe as mp
import random
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ───────────────────────────────────────────────
# Configuration (Hardcoded for notebook)
# ───────────────────────────────────────────────
# SET YOUR DESIRED CONFIGURATION HERE:
CONFIDENCE_THRESHOLD = 0.3
FLIP_SCREEN = True  # Set to True for mirror mode, False for normal

# Objects to detect (empty list = all except person)
# Examples: 
#   [] = all except person (default)
#   ["bottle", "hot dog", "cup"] = only these objects
DETECT_OBJECTS = []

# Objects to exclude (always excluded even if in DETECT_OBJECTS)
# Default excludes person
EXCLUDE_OBJECTS = ["person"]

# Show all objects including person? (overrides EXCLUDE_OBJECTS)
SHOW_ALL = False

# Display configuration
print("📓 Coin Catcher - Notebook Configuration")
print("   Confidence threshold:", CONFIDENCE_THRESHOLD)
print("   Flip screen:", "ON" if FLIP_SCREEN else "OFF")
print("   Show all objects:", "YES" if SHOW_ALL else "NO")
if DETECT_OBJECTS:
    print("   Detecting specific objects:", DETECT_OBJECTS)
else:
    print("   Detecting: All objects" + ("" if SHOW_ALL else " except person"))
if EXCLUDE_OBJECTS and not SHOW_ALL:
    print("   Excluding:", EXCLUDE_OBJECTS)

# ───────────────────────────────────────────────
# Game Settings
# ───────────────────────────────────────────────
items = ["bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
         "traffic", "light", "fire", "hydrant", "stop", "sign", "parking", "meter",
         "bench", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
         "skis", "snowboard", "sports", "ball", "kite", "baseball", "bat", "baseball",
         "glove", "skateboard", "surfboard", "tennis", "racket", "bottle", "wine",
         "glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
         "orange", "broccoli", "carrot", "hot", "dog", "pizza", "donut", "cake", "chair",
         "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
         "mouse", "remote", "keyboard", "cell", "phone", "microwave", "oven", "toaster",
         "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy", "bear",
         "hair", "drier", "toothbrush", "cell phone"]

coin_hidden_since = 0.0                  # timestamp when it was last hidden
COIN_RESPAWN_DELAY = 3.0                 # seconds
COIN_SIZE = 120                          # coin diameter in pixels
MAX_PLACEMENT_ATTEMPTS = 100             # max attempts to find non-overlapping position
COIN_SPEED = 4.0                         # pixels per frame (normal speed)
COIN_ESCAPE_SPEED = 20.0                 # pixels per frame when escaping (10x faster!)
COIN_AVOID_DISTANCE = 100                # distance at which coin starts avoiding objects
    
# Rotation settings
ROTATION_SPEED = 0.0                     # degrees per frame
ROTATION_DIRECTION_CHANGE_TIME = 120.0   # seconds between direction changes
last_direction_change_time = time.time()
current_rotation_angle = 0.0             # current rotation angle in degrees
current_rotation_direction = 1           # 1 for clockwise, -1 for counter-clockwise

# Escape state tracking
coin_is_escaping = False
escape_velocity = [0.0, 0.0]     # escape direction vector
escape_direction_set = False     # whether escape direction has been set

# ───────────────────────────────────────────────
# MediaPipe Hand Landmarker setup
# ───────────────────────────────────────────────
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

hand_landmarker = HandLandmarker.create_from_options(options)

# ───────────────────────────────────────────────
# Load images
# ───────────────────────────────────────────────
overlay_img = cv2.imread("InnoX true rock.png", cv2.IMREAD_UNCHANGED)
coin_orig = cv2.imread("InnoX coin.png", cv2.IMREAD_UNCHANGED)

if coin_orig is None:
    print("ERROR: Could not load InnoX coin.png")
    coin_img = None
else:
    coin_img = cv2.resize(coin_orig, (COIN_SIZE, COIN_SIZE))
    print("✅ Coin image loaded")

if overlay_img is None:
    print("⚠ Warning: Could not load InnoX true rock.png")

# YOLO model
model = YOLO("yolov8n.pt")

# Determine which classes to detect
def get_class_ids_to_detect(model):
    """Get YOLO class IDs based on configuration"""
    all_class_ids = list(model.names.keys())
    
    # If user specified specific objects
    if DETECT_OBJECTS:
        class_ids = []
        for obj_name in DETECT_OBJECTS:
            found = False
            for cls_id, name in model.names.items():
                if name == obj_name:
                    class_ids.append(cls_id)
                    found = True
                    print(f"✓ Added '{obj_name}' (class {cls_id}) to detection list")
                    break
            if not found:
                print(f"⚠ Warning: Object '{obj_name}' not found in YOLO classes")
        return class_ids
    
    # If user wants to see all objects (including person)
    if SHOW_ALL:
        print("✓ Detecting ALL objects (including person)")
        return all_class_ids
    
    # Default: detect all except excluded objects
    exclude_ids = []
    for obj_name in EXCLUDE_OBJECTS:
        for cls_id, name in model.names.items():
            if name == obj_name:
                exclude_ids.append(cls_id)
                print(f"✓ Excluding '{obj_name}' (class {cls_id}) from detection")
                break
    
    # Filter out excluded classes
    class_ids = [cls_id for cls_id in all_class_ids if cls_id not in exclude_ids]
    return class_ids

# Get class IDs to detect based on configuration
CLASS_IDS_TO_DETECT = get_class_ids_to_detect(model)
print(f"\n📊 Detection Summary:")
print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"   Detecting {len(CLASS_IDS_TO_DETECT)} object classes")
if CLASS_IDS_TO_DETECT:
    detected_names = [model.names[cls_id] for cls_id in CLASS_IDS_TO_DETECT[:10]]
    print(f"   First 10: {', '.join(detected_names)}")
    if len(CLASS_IDS_TO_DETECT) > 10:
        print(f"   ... and {len(CLASS_IDS_TO_DETECT) - 10} more")
print(f"   Screen flip: {'ON' if FLIP_SCREEN else 'OFF'}\n")

# Persistent state
show_coin = True
coin_position = None  # (x, y) center position of coin
coin_velocity = [0.0, 0.0]  # (dx, dy) velocity vector (normal movement)
detected_boxes = []   # list of current object bounding boxes
frame_width = 640     # will be updated from camera
frame_height = 480    # will be updated from camera

# Store rotated coin image to avoid recomputing every frame
rotated_coin_cache = None
last_rotation_angle = None

# Hand landmark connections (for manual drawing)
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),         # thumb
    (0,5), (5,6), (6,7), (7,8),         # index
    (0,9), (9,10),(10,11),(11,12),      # middle
    (0,13),(13,14),(14,15),(15,16),     # ring
    (0,17),(17,18),(18,19),(19,20),     # pinky
    (5,9), (9,13), (13,17)              # palm
]

def draw_hand_landmarks(frame, hand_landmarks_list):
    for hand_landmarks in hand_landmarks_list:
        h, w = frame.shape[:2]
        # Landmarks as dots
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
        
        # Connections as lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end   = hand_landmarks[end_idx]
            x1,y1 = int(start.x * w), int(start.y * h)
            x2,y2 = int(end.x   * w), int(end.y   * h)
            cv2.line(frame, (x1,y1), (x2,y2), (255, 0, 255), 2)
    return frame

def check_coin_overlap(coin_pos, boxes, frame_shape):
    """Check if coin overlaps with any detected object"""
    h, w = frame_shape[:2]
    cx, cy = coin_pos
    coin_radius = COIN_SIZE // 2
    
    # Check if coin is within frame bounds
    if (cx - coin_radius < 0 or cx + coin_radius >= w or
        cy - coin_radius < 0 or cy + coin_radius >= h):
        return True
    
    # Check overlap with detected objects
    for box in boxes:
        x1, y1, x2, y2 = box
        # Simple rectangle-circle collision check
        closest_x = max(x1, min(cx, x2))
        closest_y = max(y1, min(cy, y2))
        distance = np.sqrt((cx - closest_x)**2 + (cy - closest_y)**2)
        
        if distance < coin_radius:
            return True
    
    return False

def check_coin_far_from_all_objects(coin_pos, boxes, min_distance=COIN_SIZE):
    """Check if coin is at least min_distance away from all objects"""
    cx, cy = coin_pos
    
    for box in boxes:
        x1, y1, x2, y2 = box
        # Calculate center of object
        obj_cx = (x1 + x2) / 2
        obj_cy = (y1 + y2) / 2
        
        # Calculate distance to object center
        distance = np.sqrt((cx - obj_cx)**2 + (cy - obj_cy)**2)
        
        # If too close to any object, not safe yet
        if distance < min_distance:
            return False
    
    return True

def place_coin_randomly(frame, boxes):
    """Try to place coin at random position without overlapping"""
    h, w = frame.shape[:2]
    coin_radius = COIN_SIZE // 2
    
    for attempt in range(MAX_PLACEMENT_ATTEMPTS):
        # Generate random position
        cx = random.randint(coin_radius, w - coin_radius - 1)
        cy = random.randint(coin_radius, h - coin_radius - 1)
        
        if not check_coin_overlap((cx, cy), boxes, frame.shape):
            # Initialize random velocity
            angle = random.uniform(0, 2 * np.pi)
            coin_velocity[0] = COIN_SPEED * np.cos(angle)
            coin_velocity[1] = COIN_SPEED * np.sin(angle)
            return (cx, cy)
    
    # If no valid position found after max attempts, return center
    print("Warning: Could not find non-overlapping position after max attempts")
    angle = random.uniform(0, 2 * np.pi)
    coin_velocity[0] = COIN_SPEED * np.cos(angle)
    coin_velocity[1] = COIN_SPEED * np.sin(angle)
    return (w // 2, h // 2)

def update_coin_position(frame, boxes):
    """Update coin position with movement, collision avoidance, and escape behavior"""
    global coin_position, coin_velocity, show_coin
    global current_rotation_direction, last_direction_change_time
    global coin_is_escaping, escape_velocity, escape_direction_set
    
    if coin_position is None:
        return coin_position
    
    h, w = frame.shape[:2]
    cx, cy = coin_position
    coin_radius = COIN_SIZE // 2
    
    # Check if coin is currently inside any object
    is_inside_object = check_coin_overlap((cx, cy), boxes, frame.shape)
    
    # FEATURE 2: Escape behavior when inside an object
    if is_inside_object:
        if not coin_is_escaping:
            # Just entered an object - start escape mode
            coin_is_escaping = True
            escape_direction_set = False
            print("Coin entered object! Starting escape sequence...")
        
        if not escape_direction_set:
            # Generate random escape direction
            escape_angle = random.uniform(0, 2 * np.pi)
            escape_velocity[0] = np.cos(escape_angle)
            escape_velocity[1] = np.sin(escape_angle)
            escape_direction_set = True
            print(f"Escape direction set: {escape_angle:.2f} radians")
        
        # Move at escape speed in the escape direction
        dx = escape_velocity[0] * COIN_ESCAPE_SPEED
        dy = escape_velocity[1] * COIN_ESCAPE_SPEED
        
    else:
        # Coin is NOT inside any object
        if coin_is_escaping:
            # Just escaped from object
            coin_is_escaping = False
            escape_direction_set = False
            print("Coin escaped from object! Resuming normal movement.")
        
        # Normal movement with avoidance
        dx, dy = coin_velocity
        
        # Avoid detected objects (non-person objects)
        avoid_vector = [0.0, 0.0]
        for box in boxes:
            x1, y1, x2, y2 = box
            # Calculate center of object
            obj_cx = (x1 + x2) / 2
            obj_cy = (y1 + y2) / 2
            
            # Calculate distance to object
            distance = np.sqrt((cx - obj_cx)**2 + (cy - obj_cy)**2)
            
            if distance < COIN_AVOID_DISTANCE:
                # Calculate repulsion vector (away from object)
                repulsion_strength = (COIN_AVOID_DISTANCE - distance) / COIN_AVOID_DISTANCE
                avoid_vector[0] += (cx - obj_cx) * repulsion_strength * 0.1
                avoid_vector[1] += (cy - obj_cy) * repulsion_strength * 0.1
        
        # Apply avoidance
        dx += avoid_vector[0]
        dy += avoid_vector[1]
        
        # Normalize velocity to maintain normal speed
        speed = np.sqrt(dx**2 + dy**2)
        if speed > 0:
            dx = (dx / speed) * COIN_SPEED
            dy = (dy / speed) * COIN_SPEED
        
        # Update the stored normal velocity
        coin_velocity[0] = dx
        coin_velocity[1] = dy
    
    # Update position based on current mode
    new_cx = cx + dx
    new_cy = cy + dy
    
    # Bounce off walls
    # Check for wall collisions and turn 120° clockwise
    bounced = False
    if new_cx - coin_radius < 0 or new_cx + coin_radius >= w:
        # Turn 120 degrees clockwise (2π/3 radians)
        if coin_is_escaping:
            # During escape: rotate escape velocity by 120°
            angle = np.arctan2(escape_velocity[1], escape_velocity[0])
            angle += 2 * np.pi / 3  # Add 120° clockwise
            escape_velocity[0] = np.cos(angle)
            escape_velocity[1] = np.sin(angle)
            dx = escape_velocity[0] * COIN_ESCAPE_SPEED
            dy = escape_velocity[1] * COIN_ESCAPE_SPEED
        else:
            # Normal mode: rotate velocity by 120°
            angle = np.arctan2(dy, dx)
            angle += 2 * np.pi / 3  # Add 120° clockwise
            dx = COIN_SPEED * np.cos(angle)
            dy = COIN_SPEED * np.sin(angle)
            # Update stored velocity for normal movement
            coin_velocity[0] = dx
            coin_velocity[1] = dy
        
        # Keep coin within bounds
        new_cx = max(coin_radius, min(new_cx, w - coin_radius))
        bounced = True
    
    if new_cy - coin_radius < 0 or new_cy + coin_radius >= h:
        # Turn 120 degrees clockwise (2π/3 radians)
        if coin_is_escaping:
            # During escape: rotate escape velocity by 120°
            angle = np.arctan2(escape_velocity[1], escape_velocity[0])
            angle += 2 * np.pi / 3  # Add 120° clockwise
            escape_velocity[0] = np.cos(angle)
            escape_velocity[1] = np.sin(angle)
            dx = escape_velocity[0] * COIN_ESCAPE_SPEED
            dy = escape_velocity[1] * COIN_ESCAPE_SPEED
        else:
            # Normal mode: rotate velocity by 120°
            angle = np.arctan2(dy, dx)
            angle += 2 * np.pi / 3  # Add 120° clockwise
            dx = COIN_SPEED * np.cos(angle)
            dy = COIN_SPEED * np.sin(angle)
            # Update stored velocity for normal movement
            coin_velocity[0] = dx
            coin_velocity[1] = dy
        
        # Keep coin within bounds
        new_cy = max(coin_radius, min(new_cy, h - coin_radius))
        bounced = True
    
    # If we bounced during escape, update escape velocity magnitude
    if bounced and coin_is_escaping:
        speed = np.sqrt(dx**2 + dy**2)
        if speed > 0:
            dx = (dx / speed) * COIN_ESCAPE_SPEED
            dy = (dy / speed) * COIN_ESCAPE_SPEED
    
    # Check for collision after movement
    if check_coin_overlap((new_cx, new_cy), boxes, frame.shape) and not coin_is_escaping:
        # If colliding during normal movement, change direction
        angle = random.uniform(0, 2 * np.pi)
        coin_velocity[0] = COIN_SPEED * np.cos(angle)
        coin_velocity[1] = COIN_SPEED * np.sin(angle)
        # Move away from collision
        new_cx = cx + coin_velocity[0] * 2
        new_cy = cy + coin_velocity[1] * 2
    
    return (new_cx, new_cy)

def get_rotated_coin(angle_degrees):
    """Get rotated coin image with caching to improve performance"""
    global rotated_coin_cache, last_rotation_angle, coin_img
    
    # Return cached image if angle hasn't changed significantly
    if (rotated_coin_cache is not None and last_rotation_angle is not None and 
        abs(angle_degrees - last_rotation_angle) < 0.1):
        return rotated_coin_cache
    
    # Get image dimensions and center
    h, w = coin_img.shape[:2]
    center = (w // 2, h // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Calculate new bounding dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Adjust rotation matrix to account for translation
    rotation_matrix[0, 2] += new_w / 2 - center[0]
    rotation_matrix[1, 2] += new_h / 2 - center[1]
    
    # Rotate the image
    if coin_img.shape[2] == 4:  # RGBA image
        rotated = cv2.warpAffine(coin_img, rotation_matrix, (new_w, new_h), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    else:  # RGB image
        rotated = cv2.warpAffine(coin_img, rotation_matrix, (new_w, new_h), 
                                flags=cv2.INTER_LINEAR)
    
    # Cache the result
    rotated_coin_cache = rotated
    last_rotation_angle = angle_degrees
    
    return rotated

def detect_objects(frame):
    global show_coin, coin_hidden_since, coin_position, detected_boxes, coin_velocity
    global frame_width, frame_height, current_rotation_angle, current_rotation_direction
    global last_direction_change_time, coin_is_escaping
    
    # Update frame dimensions
    h, w = frame.shape[:2]
    frame_width = w
    frame_height = h
    
    # Auto-respawn after delay
    current_time = time.time()
    if not show_coin and (current_time - coin_hidden_since >= COIN_RESPAWN_DELAY):
        show_coin = True
        coin_position = None  # Will be placed randomly
        coin_is_escaping = False  # Reset escape state
        print("Coin respawned!")
    
    # FEATURE 1: Update rotation direction every 5 seconds
    if current_time - last_direction_change_time >= ROTATION_DIRECTION_CHANGE_TIME:
        current_rotation_direction *= -1  # Flip direction
        last_direction_change_time = current_time
        direction_text = "clockwise" if current_rotation_direction == 1 else "counter-clockwise"
        print(f"Coin rotation direction changed to {direction_text}")
    
    # Update rotation angle
    current_rotation_angle += current_rotation_direction * ROTATION_SPEED
    # Keep angle in 0-360 range for readability
    current_rotation_angle %= 360
    
    # Run YOLO with class filtering
    results = model(frame, conf=CONFIDENCE_THRESHOLD, 
                   classes=CLASS_IDS_TO_DETECT, verbose=False)
    
    detected_objects = []
    current_boxes = []
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0].item()
            label = model.names[cls_id]
            
            detected_objects.append(label)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_boxes.append((x1, y1, x2, y2))
            
            # Only overlay on non-person objects
            if overlay_img is not None and label in items:
                box_w = x2 - x1
                box_h = y2 - y1
                target_size = min(box_w, box_h)
                orig_h, orig_w = overlay_img.shape[:2]
                scale = target_size / max(orig_w, orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                
                resized = cv2.resize(overlay_img, (new_w, new_h), cv2.INTER_AREA)
                
                x_offset = x1 + (box_w - new_w) // 2
                y_offset = y1 + (box_h - new_h) // 2
                
                y1_roi = max(0, y_offset)
                y2_roi = min(h, y_offset + new_h)
                x1_roi = max(0, x_offset)
                x2_roi = min(w, x_offset + new_w)
                
                if y1_roi < y2_roi and x1_roi < x2_roi:
                    
                    # Convert all slice indices to integers
                    y1_roi_int = int(y1_roi)
                    y2_roi_int = int(y2_roi)
                    x1_roi_int = int(x1_roi)
                    x2_roi_int = int(x2_roi)
                    y_offset_int = int(y_offset)
                    x_offset_int = int(x_offset)
                    
                    crop = resized[y1_roi_int - y_offset_int:y2_roi_int - y_offset_int,
                                   x1_roi_int - x_offset_int:x2_roi_int - x_offset_int]
                    roi = frame[y1_roi_int:y2_roi_int, x1_roi_int:x2_roi_int]
                    
                    if resized.shape[2] == 4:
                        alpha = crop[:, :, 3] / 255.0
                        for c in range(3):
                            roi[:, :, c] = alpha * crop[:, :, c] + (1 - alpha) * roi[:, :, c]
                    else:
                        roi[:] = crop
                    
                    frame[y1_roi_int:y2_roi_int, x1_roi_int:x2_roi_int] = roi
            
            # Draw bounding boxes (all objects here are non-person)
            if coin_is_escaping:
                color = (0, 100, 255)  # Orange-red for "danger" mode
                thickness = 3
            else:
                color = (0, 0, 255)    # Normal red
                thickness = 2
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    
    # Update detected boxes (only non-person objects)
    detected_boxes = current_boxes
    
    # ── Coin logic ──
    coin_placed_this_frame = False
    
    if show_coin and coin_img is not None:
        # Check if coin needs initial placement
        if coin_position is None:
            coin_position = place_coin_randomly(frame, detected_boxes)
            coin_is_escaping = False  # Reset escape state on respawn
            print(f"Coin placed at ({int(coin_position[0])}, {int(coin_position[1])})")
        
        # Update coin position with movement
        coin_position = update_coin_position(frame, detected_boxes)
        
        # Get rotated coin image (FEATURE 1)
        rotated_coin = get_rotated_coin(current_rotation_angle)
        rh, rw = rotated_coin.shape[:2]
        
        # Draw rotated coin at current position
        cx, cy = coin_position
        
        x_offset = cx - rw // 2
        y_offset = cy - rh // 2
        
        y1r = max(0, y_offset)
        y2r = min(h, y_offset + rh)
        x1r = max(0, x_offset)
        x2r = min(w, x_offset + rw)
        
        if y1r < y2r and x1r < x2r:
            
            # Convert all slice indices to integers
            y1r_int = int(y1r)
            y2r_int = int(y2r)
            x1r_int = int(x1r)
            x2r_int = int(x2r)
            y_offset_int = int(y_offset)
            x_offset_int = int(x_offset)
            
            crop = rotated_coin[y1r_int - y_offset_int:y2r_int - y_offset_int,
                                x1r_int - x_offset_int:x2r_int - x_offset_int]
            roi = frame[y1r_int:y2r_int, x1r_int:x2r_int]
            
            if rotated_coin.shape[2] == 4:
                alpha = crop[:, :, 3] / 255.0
                for c in range(3):
                    roi[:, :, c] = alpha * crop[:, :, c] + (1 - alpha) * roi[:, :, c]
            else:
                roi[:] = crop
            
            frame[y1r_int:y2r_int, x1r_int:x2r_int] = roi
        
        coin_placed_this_frame = True
    
    # ── Hand detection & catch ──
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    detection_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    
    if show_coin and coin_placed_this_frame and coin_position is not None:
        cx, cy = coin_position
        coin_radius_px = COIN_SIZE // 2
        
        if detection_result.hand_landmarks:
            for hand_lm in detection_result.hand_landmarks:
                # Check multiple finger tips for better catch detection
                tips_to_check = [8, 12, 16, 20]  # index, middle, ring, pinky tips
                
                for tip_idx in tips_to_check:
                    tip = hand_lm[tip_idx]
                    hx = int(tip.x * w)
                    hy = int(tip.y * h)
                    
                    dist = np.hypot(hx - cx, hy - cy)
                    if dist < coin_radius_px + 20:  # Catch radius
                        show_coin = False
                        coin_position = None
                        coin_hidden_since = time.time()
                        coin_is_escaping = False  # Reset escape state
                        print("Coin caught! Respawning in 3 seconds...")
                        break
                
                if not show_coin:
                    break
            
            # Draw hands
            frame = draw_hand_landmarks(frame, detection_result.hand_landmarks)
    
    # Display coin velocity vector and rotation info (for debugging)
    if coin_position is not None and show_coin:
        cx, cy = coin_position
        if coin_is_escaping:
            dx, dy = escape_velocity[0] * COIN_ESCAPE_SPEED, escape_velocity[1] * COIN_ESCAPE_SPEED
            color = (0, 255, 255)  # Cyan for escape mode
            thickness = 3
        else:
            dx, dy = coin_velocity
            color = (255, 255, 0)  # Yellow for normal mode
            thickness = 2
            
        end_x = int(cx + dx * 2)  # Scale for visualization
        end_y = int(cy + dy * 2)
        cv2.arrowedLine(frame, (int(cx), int(cy)), (end_x, end_y), color, thickness)
        
        # Display rotation info
        direction_text = "CW" if current_rotation_direction == 1 else "CCW"
        mode_text = "ESCAPE" if coin_is_escaping else "NORMAL"
        rotation_info = f"Rot: {current_rotation_angle:.1f}° ({direction_text}) | Mode: {mode_text}"
        
        # Use different text color for escape mode
        text_color = (0, 255, 255) if coin_is_escaping else (255, 200, 0)
        cv2.putText(frame, rotation_info, (w - 350, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    return frame, detected_objects

def main():
    global frame_width, frame_height
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    window_name = "Coin Catcher - Avoid objects, catch with hand! Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Starting... Coin will move and avoid objects. Catch it with your hand!")
    print("Enhanced Features:")
    print("1. Coin rotates slowly (0.5°/frame) and changes direction every 5 seconds")
    print("2. When inside object: moves 20 units/frame in random direction until fully outside")
    print("3. Visual feedback: Escape mode = cyan arrow, orange-red boxes")
    print(f"4. Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("Press 'q' or Esc to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply screen flip if requested
        if FLIP_SCREEN:
            frame = cv2.flip(frame, 1)
        
        frame, detected = detect_objects(frame)
        
        # Display stats
        mode_text = "ESCAPE!" if coin_is_escaping else "Active"
        stats_text = f"Objects: {len(detected_boxes)}  Coin: {mode_text if show_coin else 'Respawning'}"
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if not show_coin:
            respawn_time = COIN_RESPAWN_DELAY - (time.time() - coin_hidden_since)
            if respawn_time > 0:
                respawn_text = f"Respawn in: {respawn_time:.1f}s"
                cv2.putText(frame, respawn_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, frame)
        
        if detected:
            # Person is already filtered out, so just print all detected objects
            print("Detected objects:", detected)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()