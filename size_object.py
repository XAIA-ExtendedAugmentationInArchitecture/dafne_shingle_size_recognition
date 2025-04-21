import numpy as np
import cv2
import pyrealsense2 as rs
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils

# Setup ArUco
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
arucoParams = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(ARUCO_DICT, arucoParams)


# Configure RealSense pipeline (color + depth)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()


def get_real_world_distance(p1, p2, depth_frame):
    height, width = depth_frame.get_height(), depth_frame.get_width()

    # Clamp coordinates to frame size
    x1, y1 = int(np.clip(p1[0], 0, width - 1)), int(np.clip(p1[1], 0, height - 1))
    x2, y2 = int(np.clip(p2[0], 0, width - 1)), int(np.clip(p2[1], 0, height - 1))

    depth1 = get_avg_depth(x1, y1, depth_frame)
    depth2 = get_avg_depth(x2, y2, depth_frame)

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    pt1_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [x1, y1], depth1)
    pt2_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [x2, y2], depth2)

    dist = np.linalg.norm(np.array(pt1_3d) - np.array(pt2_3d))
    return dist * 100  # cm


def get_avg_depth(x, y, depth_frame, k=3):
    """Sample a kxk neighborhood and average depth values."""
    h, w = depth_frame.get_height(), depth_frame.get_width()
    x, y = int(x), int(y)
    xs = np.clip(np.arange(x - k//2, x + k//2 + 1), 0, w-1)
    ys = np.clip(np.arange(y - k//2, y + k//2 + 1), 0, h-1)
    depths = [depth_frame.get_distance(ix, iy) for ix in xs for iy in ys]
    depths = [d for d in depths if d > 0]  # remove invalids
    return np.mean(depths) if depths else 0


try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        display = image.copy()

        # --- ArUco detection (optional) ---
        corners, ids, _ = aruco_detector.detectMarkers(image)
        if ids is not None:
            for marker_corners in corners:
                cv2.polylines(display, [np.int32(marker_corners)], True, (0, 255, 0), 2)
            cv2.putText(display, "Marker Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No ArUco marker", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Object detection ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edged = cv2.Canny(blur, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = [c for c in cnts if cv2.contourArea(c) > 100]

        if not cnts:
            cv2.imshow("RealSense", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        (cnts, _) = contours.sort_contours(cnts)

        for cnt in cnts:
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box

            # Draw box
            cv2.drawContours(display, [box.astype("int")], -1, (0, 0, 255), 2)

            # Measure real-world distances using depth
            width = get_real_world_distance(tl, tr, depth_frame)
            height = get_real_world_distance(tr, br, depth_frame)

            # Draw dimensions
            mid_pt_horizontal = (int((tl[0] + tr[0]) / 2), int((tl[1] + tr[1]) / 2))
            mid_pt_vertical = (int((tr[0] + br[0]) / 2), int((tr[1] + br[1]) / 2))

            cv2.putText(display, "{:.1f}cm".format(width),
                        (mid_pt_horizontal[0] - 15, mid_pt_horizontal[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(display, "{:.1f}cm".format(height),
                        (mid_pt_vertical[0] + 10, mid_pt_vertical[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.imshow("RealSense", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
