#!/usr/bin/env python3
"""
gemini_vision_node.py

ROS1 Noetic node that:
  - Subscribes to /usb_cam/image_raw (sensor_msgs/Image)
  - Sends the latest frame to Gemini with a navigation prompt
  - Uses Gemini function calling to get a structured list of Twist + duration commands
  - Executes each command sequentially on /cmd_vel (geometry_msgs/Twist)
  - Publishes a status string to /gemini_vision/response (std_msgs/String)

Compatible with Python 3.7 on Debian 10 / ROS Noetic.

Dependencies:
  sudo apt-get install python3-opencv
  pip3 install "httpx==0.23.3"

Environment variable:
  export GEMINI_API_KEY="AIza..."

Usage:
  rosrun <your_package> gemini_vision_node.py
"""

import base64
import json
import os
import threading
import time

import cv2
import httpx
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID       = "gemini-2.0-flash"
MAX_TOKENS     = 512
JPEG_QUALITY   = 75
QUERY_INTERVAL = 30.0   # seconds between vision+motion cycles
LINEAR_CLAMP   = 0.3    # m/s  — max forward/back speed (safety limit)
ANGULAR_CLAMP  = 0.8    # rad/s — max turn speed (safety limit)
MAX_DURATION   = 5.0    # seconds — max any single command can run

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

PROMPT = (
    "You are controlling a mobile robot. Look at the image and identify the "
    "nearest object. Generate a sequence of movement commands to drive the "
    "robot toward that object and stop just in front of it. "
    "You MUST call the move_robot function with your commands. "
    "Use only forward motion (positive linear_x) and turns (angular_z). "
    "Keep speeds gentle: linear_x <= 0.3 m/s, angular_z <= 0.8 rad/s. "
    "Each command duration should be between 0.5 and 5.0 seconds."
)

# ---------------------------------------------------------------------------
# Gemini function/tool definition
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "function_declarations": [
            {
                "name": "move_robot",
                "description": (
                    "Send a sequence of velocity commands to a differential "
                    "drive robot to navigate toward the nearest visible object."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "description": "Ordered list of Twist commands with durations.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "linear_x": {
                                        "type": "number",
                                        "description": (
                                            "Forward velocity in m/s. "
                                            "Positive = forward, negative = backward."
                                        ),
                                    },
                                    "angular_z": {
                                        "type": "number",
                                        "description": (
                                            "Rotation velocity in rad/s. "
                                            "Positive = left turn, negative = right turn."
                                        ),
                                    },
                                    "duration_sec": {
                                        "type": "number",
                                        "description": "How long to apply this command in seconds.",
                                    },
                                    "description": {
                                        "type": "string",
                                        "description": "Brief human-readable note, e.g. 'turn left to face chair'.",
                                    },
                                },
                                "required": ["linear_x", "angular_z", "duration_sec"],
                            },
                        },
                        "target_description": {
                            "type": "string",
                            "description": "What object the robot is moving toward.",
                        },
                    },
                    "required": ["commands", "target_description"],
                },
            }
        ]
    }
]


class GeminiVisionNode:
    def __init__(self):
        rospy.init_node("gemini_vision_node", anonymous=False)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            rospy.logfatal("GEMINI_API_KEY environment variable is not set. Exiting.")
            raise SystemExit(1)

        self._api_url = GEMINI_API_URL.format(model=MODEL_ID)
        self._headers = {
            "Content-Type":   "application/json",
            "x-goog-api-key": api_key,
        }

        self._bridge        = CvBridge()
        self._frame_lock    = threading.Lock()
        self._motion_lock   = threading.Lock()  # prevents overlapping motion sequences
        self._latest_frame  = None
        self._executing     = False             # True while a motion sequence is running

        # Publishers
        self._cmd_pub      = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._response_pub = rospy.Publisher("/gemini_vision/response", String, queue_size=10)

        # Subscriber
        rospy.Subscriber(
            "/usb_cam/image_raw", Image, self._image_callback,
            queue_size=1, buff_size=2 ** 24,
        )

        # Timer
        rospy.Timer(rospy.Duration(QUERY_INTERVAL), self._timer_callback)

        rospy.loginfo("gemini_vision_node ready. Cycle every %.1f s.", QUERY_INTERVAL)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _image_callback(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._frame_lock:
                self._latest_frame = frame
        except Exception as exc:
            rospy.logwarn("cv_bridge error: %s", exc)

    def _timer_callback(self, _event):
        # Don't query while a motion sequence is still running
        if self._executing:
            rospy.loginfo("Still executing previous motion sequence — skipping query.")
            return

        with self._frame_lock:
            frame = self._latest_frame

        if frame is None:
            rospy.loginfo_throttle(10, "No image yet — skipping.")
            return

        t = threading.Thread(target=self._query_and_execute, args=(frame,), daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Gemini API — function calling
    # ------------------------------------------------------------------

    def _encode_frame(self, frame):
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def _query_and_execute(self, frame):
        try:
            b64_image = self._encode_frame(frame)

            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": b64_image,
                                }
                            },
                            {"text": PROMPT},
                        ]
                    }
                ],
                "tools": TOOLS,
                "generationConfig": {
                    "maxOutputTokens": MAX_TOKENS,
                    "temperature": 0.2,  # low temp for consistent structured output
                },
            }

            rospy.loginfo("Sending image to Gemini...")

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    self._api_url,
                    headers=self._headers,
                    content=json.dumps(payload),
                )

            if resp.status_code != 200:
                rospy.logerr("API error %d: %s", resp.status_code, resp.text[:300])
                return

            data = resp.json()
            self._handle_response(data)

        except httpx.TimeoutException:
            rospy.logerr("Gemini API request timed out.")
        except Exception as exc:
            rospy.logerr("Unexpected error in _query_and_execute: %s", exc)

    def _handle_response(self, data):
        """Extract the function call from Gemini's response and execute it."""
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as exc:
            rospy.logerr("Unexpected response structure: %s", exc)
            return

        for part in parts:
            if "functionCall" in part:
                fn        = part["functionCall"]
                fn_name   = fn.get("name")
                fn_args   = fn.get("args", {})

                if fn_name != "move_robot":
                    rospy.logwarn("Unexpected function call: %s", fn_name)
                    continue

                target   = fn_args.get("target_description", "unknown object")
                commands = fn_args.get("commands", [])

                rospy.loginfo("Target: %s | %d command(s) received.", target, len(commands))
                self._response_pub.publish(
                    String(data="Moving toward: {}  ({} steps)".format(target, len(commands)))
                )

                self._execute_commands(commands)
                return

        # If we get here Gemini returned text instead of a function call
        text_parts = [p.get("text", "") for p in parts if "text" in p]
        rospy.logwarn("No function call in response. Text: %s", " ".join(text_parts))

    # ------------------------------------------------------------------
    # Motion execution
    # ------------------------------------------------------------------

    def _clamp(self, value, limit):
        return max(-limit, min(limit, value))

    def _execute_commands(self, commands):
        """Run each Twist command in sequence, then stop."""
        with self._motion_lock:
            self._executing = True
            try:
                for i, cmd in enumerate(commands):
                    if rospy.is_shutdown():
                        break

                    linear_x   = self._clamp(float(cmd.get("linear_x",   0.0)), LINEAR_CLAMP)
                    angular_z  = self._clamp(float(cmd.get("angular_z",  0.0)), ANGULAR_CLAMP)
                    duration   = min(float(cmd.get("duration_sec", 1.0)), MAX_DURATION)
                    desc       = cmd.get("description", "")

                    rospy.loginfo(
                        "Step %d/%d: lin_x=%.2f  ang_z=%.2f  dur=%.2fs  [%s]",
                        i + 1, len(commands), linear_x, angular_z, duration, desc,
                    )

                    twist = Twist()
                    twist.linear.x  = linear_x
                    twist.angular.z = angular_z

                    start = time.time()
                    rate  = rospy.Rate(20)  # 20 Hz publish rate while command is active
                    while (time.time() - start) < duration and not rospy.is_shutdown():
                        self._cmd_pub.publish(twist)
                        rate.sleep()

            finally:
                # Always send a stop command when done or on error
                self._cmd_pub.publish(Twist())
                rospy.loginfo("Motion sequence complete — robot stopped.")
                self._executing = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        node = GeminiVisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
