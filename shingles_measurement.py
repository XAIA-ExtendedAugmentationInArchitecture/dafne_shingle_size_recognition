import time
import numpy as np
import cv2
import pyrealsense2 as rs
import paho.mqtt.client as mqtt
import math
import json
import random
import os

DEBUG = True  # Set to False to disable DEBUG mode
# Global variables for MQTT control
selected_action = "idle"
attempt = 0              # Attempt number for the current action
current_ids = []         # Current object id received
last_measurement = None   # Stored as (mean_width, mean_height) in cm
last_ids = []              # Object id that was last computed
last_box_norm = None      # Normalized bounding box corners of last measurement
sent_log = []  # Store all sent MQTT messages for logging


# MQTT topics
ACTION_TOPIC = "/dafne/material_registration/actions"
RESULT_TOPIC = "/dafne/material_registration/result"

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    print("MQTT Connected, result code:", rc)
    client.subscribe(ACTION_TOPIC)

def on_message(client, userdata, msg):
    global selected_action, current_ids, attempt
    
    # Expect payload in the format {"action": "compute", "ids": [123, 456], "attempt": 2}
    raw = msg.payload.decode("utf-8").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("Invalid JSON received.")
        return

    cmd = data.get("action")
    if cmd != "compute":
        print("Ignoring non-compute command.")
        return

    # 3) Validate 'id' field as a list of two ints
    ids = data.get("ids")
    if (
        not isinstance(ids, list)
        or len(ids) != 2
        or not all(isinstance(i, int) for i in ids)
    ):
        print("Invalid 'id' field; expected a list of two integers.")
        return

    # 4) Validate 'attempt' as an integer
    att = data.get("attempt")
    if not isinstance(att, int):
        print("Invalid 'attempt' field; expected an integer.")
        return

    # 5) All good—store and report
    selected_action = cmd
    current_ids = ids
    attempt = att
    print(f"Received command '{cmd}' with id {ids} (attempt {att})")


mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect("broker.emqx.io", 1883, 60)  # Adjust as needed
mqtt_client.loop_start()

# --- Utility: Average depth over a small window ---
def get_average_depth(x, y, depth_frame, kernel_size=5):
    half = kernel_size // 2
    depth_values = []
    for i in range(x - half, x + half + 1):
        for j in range(y - half, y + half + 1):
            depth = depth_frame.get_distance(i, j)
            if depth > 0:
                depth_values.append(depth)
    if depth_values:
        return np.median(depth_values)
    else:
        return depth_frame.get_distance(x, y)
    
def save_log_immediately():
    if sent_log:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "latest_log.json")
        with open(log_path, "w") as f:
            json.dump(sent_log, f, indent=2)


# Smoothing parameters for dimensions
alpha = 0.2  # smoothing factor (0 = very smooth, 1 = no smoothing)
smoothed_width = None
smoothed_height = None

# Threshold for position change (in normalized coordinates, 0–1)
position_change_threshold = 0.1  # 10% of the ROI size

# --- Initialize RealSense ---
if not DEBUG:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

# --- Prepare ArUco Detection ---
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()
working_corners = {}  # To store ROI marker corners

# --- Process a single frame ---
def process_frame():
    global working_corners, smoothed_width, smoothed_height
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None, None, None, None, None
    color_image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # --- ArUco Detection (ROI) ---
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=arucoParams)
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if marker_id == 0:
                working_corners['top_left'] = marker_corners[0]
            elif marker_id == 25:
                working_corners['bottom_right'] = marker_corners[0]
        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
    
    # Proceed only if both markers were detected.
    if 'top_left' in working_corners and 'bottom_right' in working_corners:
        tl = working_corners['top_left'][0].astype(int)
        br = working_corners['bottom_right'][2].astype(int)
        
        # --- Create ROI mask ---
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        mask = np.zeros(thresh.shape, dtype=np.uint8)
        mask[tl[1]:br[1], tl[0]:br[0]] = 255
        roi_thresh = cv2.bitwise_and(thresh, mask)
        contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                largest_contour = cnt
        
        if largest_contour is not None:
            # Use rotated bounding box for alignment.
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(color_image, [box], 0, (255, 0, 0), 2)
            
            # --- Compute 3D points for each box corner (with depth averaging) ---
            depth_frame = aligned_frames.get_depth_frame()  # re-get depth frame if needed
            box_3d = []
            for (px, py) in box:
                avg_depth = get_average_depth(px, py, depth_frame, kernel_size=5)
                pt_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], avg_depth)
                box_3d.append(pt_3d)
            
            # --- Compute dimensions from 3D corners (in cm) ---
            edge01 = np.linalg.norm(np.array(box_3d[1]) - np.array(box_3d[0]))
            edge12 = np.linalg.norm(np.array(box_3d[2]) - np.array(box_3d[1]))
            edge23 = np.linalg.norm(np.array(box_3d[3]) - np.array(box_3d[2]))
            edge30 = np.linalg.norm(np.array(box_3d[0]) - np.array(box_3d[3]))
            dim1 = (edge01 + edge23) / 2.0 * 100  # Convert to cm
            dim2 = (edge12 + edge30) / 2.0 * 100  # Convert to cm
            edge_width = min(dim1, dim2)
            edge_height = max(dim1, dim2)
            
            # --- Temporal smoothing of dimensions ---
            if smoothed_width is None:
                smoothed_width = edge_width
                smoothed_height = edge_height
            else:
                smoothed_width = alpha * edge_width + (1 - alpha) * smoothed_width
                smoothed_height = alpha * edge_height + (1 - alpha) * smoothed_height
            
            cv2.putText(color_image, f'W: {smoothed_width:.2f} cm', (box[0][0], box[0][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(color_image, f'H: {smoothed_height:.2f} cm', (box[0][0], box[0][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # --- Compute normalized box corners relative to the ROI ---
            roi_width = br[0] - tl[0]
            roi_height = br[1] - tl[1]
            normalized_box = []
            for (px, py) in box:
                nx = (px - tl[0]) / roi_width
                ny = (py - tl[1]) / roi_height
                normalized_box.append((nx, ny))
            
            return color_image, tl, br, (smoothed_width, smoothed_height), depth_frame, box, normalized_box
    return color_image, None, None, None, None, None, None

# --- Main Loop ---
try:
    while True:
        # When a compute/recompute command is received:
        if selected_action == "compute":
            print(f"[{'DEBUG' if DEBUG else 'LIVE'}] Compute for ids {current_ids}, attempt {attempt}")

            if DEBUG:
                time.sleep(1)

                warning_type = random.choices(
                    ["success", "No Measurements", "Position Unchanged"],
                    weights=[0.7, 0.2, 0.1],  # more chance for success
                    k=1
                )[0]

                category_type = "none"

                if warning_type == "success":
                    category_type = random.choice([
                        "XSmall", "Small", "Medium", "Large", "Rand",
                        "Boundary_XS_S", "Boundary_S_M", "Boundary_M_L"
                    ])
                    expected_length = 40.0
                    threshold_length = 2.0

                    # Normal categories
                    if category_type == "Rand":
                        length = random.choice([
                            random.uniform(30, 37.9),
                            random.uniform(42.1, 50)
                        ])
                        width = random.uniform(2.0, 20.0)

                    elif category_type == "XSmall":
                        length = random.uniform(38.0, 42.0)
                        width = random.uniform(3.5, 6.9)

                    elif category_type == "Small":
                        length = random.uniform(38.0, 42.0)
                        width = random.uniform(7.1, 9.9)

                    elif category_type == "Medium":
                        length = random.uniform(38.0, 42.0)
                        width = random.uniform(10.1, 12.9)

                    elif category_type == "Large":
                        length = random.uniform(38.0, 42.0)
                        width = random.uniform(13.1, 17.0)

                    # In-between boundary cases (trigger lower confidence in Unity)
                    elif category_type == "Boundary_XS_S":
                        length = random.uniform(38.0, 42.0)
                        width = random.uniform(6.8, 7.2)

                    elif category_type == "Boundary_S_M":
                        length = random.uniform(38.0, 42.0)
                        width = random.uniform(9.8, 10.2)

                    elif category_type == "Boundary_M_L":
                        length = random.uniform(38.0, 42.0)
                        width = random.uniform(12.8, 13.2)

                    # Randomize orientation
                    if random.random() < 0.5:
                        dims = [round(width, 2), round(length, 2)]
                    else:
                        dims = [round(length, 2), round(width, 2)]

                    warning = "success"

                else:
                    dims = []
                    warning = warning_type

                result = {
                    "warning": warning,
                    "ids": current_ids,
                    "attempt": attempt,
                    "detected_dimensions": dims
                }
                
                result["timestamp"] = int(time.time())
                mqtt_client.publish(RESULT_TOPIC, json.dumps(result))
                sent_log.append(result)
                save_log_immediately()
                print(f"→ Published DEBUG result ({category_type}):", result)
                selected_action = "idle"
                continue

            measurements = []
            boxes_norm = []  # list of normalized bounding boxes
            start_time = time.time()
            while time.time() - start_time < 5.0:
                frame_data = process_frame()
                if frame_data is None:
                    continue
                color_image, tl, br, dims, depth_frame, box, normalized_box = frame_data
                if dims is not None and normalized_box is not None:
                    measurements.append(dims)
                    boxes_norm.append(normalized_box)
                cv2.imshow("Object Dimensions", color_image)
                cv2.waitKey(1)
            if measurements:
                widths, heights = zip(*measurements)
                mean_width = np.mean(widths)
                mean_height = np.mean(heights)
                new_measurement = (mean_width, mean_height)
                # Average the normalized boxes (corner-wise) over the measurement period.
                new_box_norm = np.mean(np.array(boxes_norm), axis=0)  # shape: (4,2)
                print(f"Measured (id {current_ids[0]}): W: {mean_width:.2f} cm, H: {mean_height:.2f} cm")
                
                # If this compute uses different ids as the previous one, compare positions.
                if last_box_norm is not None and current_ids != last_ids:
                    diff_sum = 0
                    for (old_pt, new_pt) in zip(last_box_norm, new_box_norm):
                        diff = math.hypot(new_pt[0] - old_pt[0], new_pt[1] - old_pt[1])
                        diff_sum += diff
                    avg_diff = diff_sum / 4.0
                    print(f"Average normalized corner difference: {avg_diff:.3f}")
                    if avg_diff < position_change_threshold:
                        result = {"warning": "Position Unchanged", "ids": current_ids, "attempt": attempt, "detected_dimensions": []}
                        result_str = json.dumps(result)
                        warning_str = f"Warning: Object (id: {current_ids}) position unchanged (avg diff {avg_diff:.3f}) compared to objects (ids {last_ids}) "
                        print(warning_str)
                        result["timestamp"] = int(time.time())
                        mqtt_client.publish(RESULT_TOPIC, result_str)
                        sent_log.append(result)
                        save_log_immediately()
                        selected_action = "idle"
                        continue
                # Publish measurement if the object position is different enough.
                result = { "warning": "success", "ids": current_ids, "attempt": attempt, "detected_dimensions": [mean_width, mean_height]}
                result_str = json.dumps(result)
                result["timestamp"] = int(time.time())
                mqtt_client.publish(RESULT_TOPIC, result_str)
                sent_log.append(result)
                save_log_immediately()

                print("Publishing result:", result_str)
                # Save current measurement data.
                last_measurement = new_measurement
                last_box_norm = new_box_norm
                last_ids = current_ids
            else:
                result = {"warning": "No Measurements", "ids": current_ids, "attempt": attempt, "detected_dimensions": []}
                result_str = json.dumps(result)
                result["timestamp"] = int(time.time())
                mqtt_client.publish(RESULT_TOPIC, result_str)
                sent_log.append(result)
                save_log_immediately()

            # Reset command.
            selected_action = "idle"
    
        
        else:
            if DEBUG:
                # don’t call process_frame; just throttle the loop
                time.sleep(0.1)
                continue
            # Idle mode: simply display frames.
            frame_data = process_frame()
            if frame_data is None:
                continue
            color_image, tl, br, dims, depth_frame, box, normalized_box = frame_data
            cv2.imshow("Object Dimensions", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    if not DEBUG:
        pipeline.stop()
    cv2.destroyAllWindows()
    mqtt_client.loop_stop()
    mqtt_client.disconnect()

    if sent_log:
        timestamp = int(time.time())  # UNIX time in seconds
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"log_{timestamp}.json")
        with open(log_path, "w") as f:
            json.dump(sent_log, f, indent=2)
        print(f"Saved MQTT log to: {os.path.abspath(log_path)}")
