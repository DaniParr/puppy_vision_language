#!/usr/bin/env python3
import cv2
import rospy
import threading
import numpy as np
from datetime import datetime
from cv_bridge import CvBridge
from puppy_control.msg import Pose
from puppy_control.srv import SetRunActionName
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String

from brain import Brain

STOP_KEYWORDS = [
    "stop",
    "abort",
    "wait",
    "hold on",
]

# Motion constants
LINEAR_SPEED  = 0.1   # m/s
ANGULAR_SPEED = 0.5   # rad/s

# Vertical baseline between camera positions (standing vs lying down)
STEREO_BASELINE_M = 0.043275  # <-- fill in your measured value in meters
MARGIN = 20

CAMERA_INTRINSIC = np.matrix([  [619.063979, 0,          302.560920],
                                [0,          613.745352, 237.714934],
                                [0,          0,          1]])

class PuppyVisionLanguageNode:
    """
    Main node for the puppy vision language.
    """
    def __init__(self):

        rospy.init_node("puppy_vision_language", anonymous=False)

        self.brain = Brain()

        # Trackers
        self.last_image       = None
        self.scanned_frame    = None
        self.last_update_time = datetime.now()

        # ROS interfacing attributes
        self._bridge         = CvBridge()
        self._frame_lock     = threading.Lock()
        self._motion_lock    = threading.Lock()
        self._latest_frame   = None
        self._executing      = False
        self._stop_event     = threading.Event()

        # Publishers
        self._pose_pub = rospy.Publisher("/puppy_control/pose", Pose, queue_size=10)
        self._vel_pub  = rospy.Publisher("/puppy_move", Point, queue_size=10)

        # Service Proxies
        self._action_service = rospy.ServiceProxy("/puppy_control/runActionGroup", SetRunActionName)

        # Subscribers
        self._prompt_sub = rospy.Subscriber("/prompt", String, self._prompt_callback, queue_size=10)
        self._img_sub    = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self._image_callback,
            queue_size=1, buff_size=2 ** 24,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        with self._frame_lock:
            self._latest_frame  = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_update_time = datetime.now()

    def _prompt_callback(self, msg: String) -> None:

        prompt        = msg.data
        time_received = datetime.now()

        if any(word in prompt.lower() for word in STOP_KEYWORDS):
            rospy.loginfo("Stop keyword detected — interrupting current sequence.")
            self._stop_event.set()
            self._publish_zero_velocity()
            self._executing = False
            return

        # Kick off execution in a background thread so the callback returns immediately
        self._executing  = True
        self._stop_event.clear()
        thread = threading.Thread(
            target=self._execute_sequence,
            args=(prompt, time_received),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # Sequence execution
    # ------------------------------------------------------------------

    def _execute_sequence(self, prompt: str, time_received: datetime) -> None:
        """
        Background thread: wait for a fresh frame, query the brain,
        then dispatch each action in order.
        """
        try:
            # Wait until we have a frame newer than when the prompt arrived
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                with self._frame_lock:
                    frame_ready = (
                        self._latest_frame is not None
                        and self.last_update_time >= time_received
                    )
                if frame_ready:
                    break
                rospy.logwarn("Waiting for a fresh frame...")
                rate.sleep()

            with self._frame_lock:
                frame = self._latest_frame.copy()
                self.scanned_frame = frame
           
            # Controller debugging commands
            if prompt == "scan":
                summary = "debugging perception"
                actions = [
                    {
                        "file_name": "move_to_object",
                        "action_type": "move_to_object",
                        "scan_center_x": 0.5,
                        "scan_center_y": 0.5,
                        "scan_width": 0.25,
                        "scan_height": 0.25,
                    }
                ]
            
            elif prompt == "move":
                summary = "moving 1 meter forward"
                actions = [
                    {
                        "file_name": "move",
                        "action_type": "move",
                        "move_meters": 1.0,
                    }
                ]
            
            elif prompt == "rotate":
                summary = "rotating 90 degrees CW"
                actions = [
                    {
                        "file_name": "rotate",
                        "action_type": "rotate",
                        "rotate_radians": -1 * np.pi / 2,
                    }
                ]
            
            else:
                summary, actions = self.brain.send_request(prompt, frame)

            if not actions:
                rospy.logwarn("No actions returned from brain.")
                self._execute_action("shake_head.d6ac")
                self._execute_action("stand.d6ac")
                return

            rospy.loginfo("Summary: %s | %d action(s)", summary, len(actions))

            for action in actions:

                if self._stop_event.is_set():
                    rospy.loginfo("Stop event detected — aborting action sequence.")
                    break

                file_name   = action.get("file_name", "")
                action_type = action.get("action_type", "default")
                description = action.get("description", "")

                rospy.loginfo(
                    "Dispatching — file: %s | type: %s | note: %s",
                    file_name, action_type, description,
                )

                if action_type == "scan" or action_type == "move_to_object":
                    cx = action.get("scan_center_x")
                    cy = action.get("scan_center_y")
                    w  = action.get("scan_width")
                    h  = action.get("scan_height")
                    if None in (cx, cy, w, h):
                        rospy.logwarn("Scan action missing bounding box fields, skipping.")
                        continue
                    self._execute_scan(file_name, cx, cy, w, h, action_type == "move_to_object")

                elif action_type == "move":
                    meters = action.get("move_meters")
                    if meters is None:
                        rospy.logwarn("Move action missing move_meters, skipping.")
                        continue
                    self._execute_move(file_name, meters)

                elif action_type == "rotate":
                    radians = action.get("rotate_radians")
                    if radians is None:
                        rospy.logwarn("Rotate action missing rotate_radians, skipping.")
                        continue
                    self._execute_rotate(file_name, radians)

                else:
                    self._execute_action(file_name)
                    self._execute_action("stand.d6ac")

        except Exception as exc:
            rospy.logerr("Error in execute_sequence: %s", exc)
            self._execute_action("pee.d6ac")
            self._execute_action("stand.d6ac")

        finally:
            self._executing = False

    # ------------------------------------------------------------------
    # Action executors
    # ------------------------------------------------------------------

    def _execute_action(self, file_name: str) -> None:
        """Play a default action file via the action group service."""
        rospy.loginfo("Playing action file: %s", file_name)
        try:
            self._action_service(file_name, 1)
        except rospy.ServiceException as exc:
            rospy.logerr("Action service call failed for '%s': %s", file_name, exc)

    def _execute_scan(self, file_name: str, cx: float, cy: float, w: float, h: float, move: bool = False) -> None:
        """
        Play the action file, capture two frames at different heights to estimate
        depth via vertical disparity, then save the annotated image.
        """
        rospy.loginfo(
            "Scan — file: %s | center=(%.2f, %.2f) | size=(%.2f x %.2f)",
            file_name, cx, cy, w, h,
        )

        settle = rospy.Duration(0.5)  # wait for robot to settle after each pose

        # --- Shot 1: standing ---
        self._execute_action("stand.d6ac")
        rospy.sleep(settle)
        with self._frame_lock:
            if self._latest_frame is None:
                rospy.logwarn("No frame available for standing shot.")
                return
            frame_stand = self._latest_frame.copy()

        # --- Shot 2: lying down ---
        self._execute_action("lie_down.d6ac")
        
        rospy.sleep(settle)
        
        time_received = datetime.now()
        
        # with self._frame_lock:

        rate = rospy.Rate(10)
        while self.last_update_time <= time_received:
            rospy.loginfo("Waiting for new frame")
            rate.sleep()

        if self._latest_frame is None:
            rospy.logwarn("No frame available for lying down shot.")
            return
        
        frame_lie = self._latest_frame.copy()

        # --- Return to standing ---
        self._execute_action("stand.d6ac")

        # --- Convert normalized coords to pixels using standing frame ---
        frame_h, frame_w = frame_stand.shape[:2]
        cx_px = cx * frame_w
        cy_px = cy * frame_h
        w_px  = w  * frame_w
        h_px  = h  * frame_h

        x1 = int(cx_px - w_px / 2)
        y1 = int(cy_px - h_px / 2)
        x2 = int(cx_px + w_px / 2)
        y2 = int(cy_px + h_px / 2)

        # --- Estimate depth via FAST keypoint vertical disparity ---
        gray_stand = cv2.cvtColor(frame_stand, cv2.COLOR_BGR2GRAY)
        gray_lie   = cv2.cvtColor(frame_lie,   cv2.COLOR_BGR2GRAY)

        fast = cv2.FastFeatureDetector_create()
        kps  = fast.detect(gray_stand, None)

        # Filter keypoints to those inside or near the bounding box
        roi_kps = [
            kp for kp in kps
            if (cx_px - w_px / 2) <= kp.pt[0] <= (cx_px + w_px / 2)
            and (cy_px - h_px / 2) <= kp.pt[1] <= (cy_px + h_px / 2)
        ]

        if not roi_kps:
            rospy.logwarn("No FAST keypoints found near bounding box — cannot estimate depth.")
            depth_m      = float("inf")
            disparity_px = 0.0
        else:
            # Track keypoints from standing frame into lying frame with Lucas-Kanade
            pts_stand = np.array([kp.pt for kp in roi_kps], dtype=np.float32).reshape(-1, 1, 2)
            pts_lie, status, _ = cv2.calcOpticalFlowPyrLK(gray_stand, gray_lie, pts_stand, None)

            # Keep only successfully tracked points
            good_stand = pts_stand[status.ravel() == 1]
            good_lie   = pts_lie[status.ravel() == 1]

            if len(good_stand) == 0:
                rospy.logwarn("No points tracked successfully — cannot estimate depth.")
                depth_m      = float("inf")
                disparity_px = 0.0
            else:
                # Vertical disparity per point, take median for robustness
                disparities  = np.abs(good_lie[:, 0, 1] - good_stand[:, 0, 1])
                disparity_px = float(np.median(disparities))

                f_y = float(CAMERA_INTRINSIC[1, 1])
                if disparity_px > 0.0:
                    depth_m = (f_y * STEREO_BASELINE_M) / disparity_px
                else:
                    depth_m = float("inf")
                    rospy.logwarn("Zero disparity — could not estimate depth.")

                rospy.loginfo(
                    "FAST tracked %d/%d points | median disparity=%.2f px | depth=%.3f m",
                    len(good_stand), len(roi_kps), disparity_px, depth_m,
                )

        # --- Annotate and save ---
        annotated = frame_stand.copy()

        # Draw tracked points
        for pt in good_stand.reshape(-1, 2):
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), -1)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(annotated, (int(cx_px), int(cy_px)), 4, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            "cx={:.0f} cy={:.0f}  w={:.0f} h={:.0f}".format(cx_px, cy_px, w_px, h_px),
            (x1, max(y1 - 22, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
        cv2.putText(
            annotated,
            "depth={:.2f}m  disp={:.1f}px  pts={}".format(depth_m, disparity_px, len(good_stand)),
            (x1, max(y1 - 6, 24)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = "/tmp/scan_{}_{}.jpg".format(file_name.replace(".", "_"), timestamp)
        cv2.imwrite(save_path, annotated)
        rospy.loginfo("Scan image saved to: %s", save_path)

        if move:
            
            if depth_m == float("inf"):
                rospy.logwarn("Refuse to move without disparity...")

            f_x = float(CAMERA_INTRINSIC[0, 0])
            c_x = float(CAMERA_INTRINSIC[0, 2])
            lateral_m = ((cx_px - c_x) / f_x) * depth_m
            rospy.loginfo(f"Moving {depth_m} meters forward, {lateral_m} meters laterally...")

            move = Point()
            move.x = depth_m * np.cos(np.arcsin(lateral_m / depth_m)) 
            move.y = lateral_m

            self._vel_pub.publish(move)


    def _execute_move(self, file_name: str, meters: float) -> None:
        """
        Play the action file, then drive forward/backward the requested distance
        using a timed velocity command.
        """
        rospy.loginfo("Move — file: %s | %.3f m", file_name, meters)

        if self._stop_event.is_set():
            return

        move     = Point()
        move.x   = meters 
        
        self._vel_pub.publish(move)
        
        return

    def _execute_rotate(self, file_name: str, radians: float) -> None:
        """
        Play the action file, then rotate by the requested angle
        using a timed velocity command.
        """
        rospy.loginfo("Rotate — file: %s | %.4f rad", file_name, radians)

        if self._stop_event.is_set():
            return

        duration = abs(radians) / ANGULAR_SPEED
        move     = Point()
        move.z   = radians
        
        self._vel_pub.publish(move)

        return


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish_zero_velocity(self) -> None:
        """Publish a zero Point to halt all motion."""
        self._vel_pub.publish(Point())


if __name__ == "__main__":
    node = PuppyVisionLanguageNode()
    rospy.spin()
