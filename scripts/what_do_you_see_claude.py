#!/usr/bin/env python3
"""
claude_vision_node.py

ROS1 Noetic node that:
  - Subscribes to /usb_cam/image_raw (sensor_msgs/Image)
  - Converts the latest frame to a base64 JPEG
  - POSTs it to the Anthropic Messages API directly via httpx (no SDK needed)
  - Publishes the model's text response to /claude_vision/response (std_msgs/String)

Compatible with Python 3.7 on Debian 10 / ROS Noetic.

Dependencies (all Python 3.7 compatible):
  pip3 install httpx opencv-python-headless

Environment variable (set before running):
  export ANTHROPIC_API_KEY="sk-ant-..."

Usage:
  rosrun <your_package> claude_vision_node.py
"""

import base64
import json
import os
import threading

import cv2
import httpx
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATIC_PROMPT  = "What do you see?"
MODEL_ID       = "claude-haiku-4-5-20251001"
MAX_TOKENS     = 512
JPEG_QUALITY   = 75    # 0-100; lower = smaller payload, faster upload
QUERY_INTERVAL = 5.0   # seconds between API calls

ANTHROPIC_API_URL     = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"


class ClaudeVisionNode:
    def __init__(self):
        rospy.init_node("claude_vision_node", anonymous=False)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            rospy.logfatal("ANTHROPIC_API_KEY environment variable is not set. Exiting.")
            raise SystemExit(1)

        self._headers = {
            "x-api-key":         api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type":      "application/json",
        }

        self._bridge       = CvBridge()
        self._lock         = threading.Lock()
        self._latest_frame = None  # most recent BGR OpenCV image

        # Publisher
        self._response_pub = rospy.Publisher(
            "/claude_vision/response", String, queue_size=10
        )

        # Subscriber — queue_size=1 so we always work with the freshest frame
        rospy.Subscriber(
            "/usb_cam/image_raw", Image, self._image_callback,
            queue_size=1, buff_size=2 ** 24
        )

        # Timer — throttle API calls regardless of camera frame rate
        rospy.Timer(rospy.Duration(QUERY_INTERVAL), self._timer_callback)

        rospy.loginfo(
            "claude_vision_node ready. Querying Claude every %.1f s.", QUERY_INTERVAL
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _image_callback(self, msg):
        """Cache the latest image (no API call here — just store it)."""
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self._latest_frame = frame
        except Exception as exc:
            rospy.logwarn("cv_bridge conversion failed: %s", exc)

    def _timer_callback(self, _event):
        """Snapshot the cached frame and fire an API call in a background thread."""
        with self._lock:
            frame = self._latest_frame

        if frame is None:
            rospy.loginfo_throttle(10, "No image received yet — skipping API call.")
            return

        t = threading.Thread(target=self._query_claude, args=(frame,), daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Claude API (raw HTTP — no SDK required)
    # ------------------------------------------------------------------

    def _encode_frame(self, frame):
        """Return a base64-encoded JPEG string for the given BGR frame."""
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def _query_claude(self, frame):
        """Build the multimodal request, POST it, and publish the reply."""
        try:
            b64_image = self._encode_frame(frame)

            payload = {
                "model": MODEL_ID,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type":       "base64",
                                    "media_type": "image/jpeg",
                                    "data":       b64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": STATIC_PROMPT,
                            },
                        ],
                    }
                ],
            }

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    ANTHROPIC_API_URL,
                    headers=self._headers,
                    content=json.dumps(payload),
                )

            if resp.status_code != 200:
                rospy.logerr(
                    "API error %d: %s", resp.status_code, resp.text[:200]
                )
                return

            data = resp.json()
            reply = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    reply = block.get("text", "")
                    break

            rospy.loginfo("Claude: %s", reply)
            self._response_pub.publish(String(data=reply))

        except httpx.TimeoutException:
            rospy.logerr("Request to Anthropic API timed out.")
        except Exception as exc:
            rospy.logerr("Unexpected error in _query_claude: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        node = ClaudeVisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
