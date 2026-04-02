#!/usr/bin/env python3
"""
claude_vision_node.py

ROS1 Noetic node that:
  - Subscribes to /usb_cam/image_raw (sensor_msgs/Image)
  - Converts the latest frame to base64 JPEG
  - Sends it to Claude Haiku (claude-haiku-4-5-20251001) with a static text prompt
  - Publishes the model's response to /claude_vision/response (std_msgs/String)

Requirements (Python 3.7 compatible):
  pip3 install anthropic opencv-python-headless

Environment:
  Export your key before running:
    export ANTHROPIC_API_KEY="sk-ant-..."

Usage:
  rosrun <your_package> claude_vision_node.py
"""

import base64
import io
import os
import threading

import anthropic
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATIC_PROMPT   = "What do you see?"
MODEL_ID        = "claude-haiku-4-5-20251001"
MAX_TOKENS      = 512
JPEG_QUALITY    = 75   # lower = smaller payload, faster round-trip
QUERY_INTERVAL  = 5.0  # seconds between API calls (avoid hammering the API)


class ClaudeVisionNode:
    def __init__(self):
        rospy.init_node("claude_vision_node", anonymous=False)

        # Pull key from environment (never hard-code it)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            rospy.logfatal("ANTHROPIC_API_KEY environment variable is not set. Exiting.")
            raise SystemExit(1)

        self._client  = anthropic.Anthropic(api_key=api_key)
        self._bridge  = CvBridge()
        self._lock    = threading.Lock()
        self._latest_frame = None   # stores the most recent CV2 BGR image

        # Publisher
        self._response_pub = rospy.Publisher(
            "/claude_vision/response", String, queue_size=10
        )

        # Subscriber
        rospy.Subscriber(
            "/usb_cam/image_raw", Image, self._image_callback, queue_size=1,
            buff_size=2**24  # large buffer for raw image data
        )

        # Timer – fire API call on a fixed schedule, not on every frame
        rospy.Timer(rospy.Duration(QUERY_INTERVAL), self._timer_callback)

        rospy.loginfo("claude_vision_node started. Querying every %.1fs.", QUERY_INTERVAL)

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _image_callback(self, msg):
        """Cache the latest image (thread-safe, no API call here)."""
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self._latest_frame = frame
        except Exception as exc:
            rospy.logwarn("cv_bridge conversion failed: %s", exc)

    def _timer_callback(self, _event):
        """Grab the cached frame and send to Claude in a background thread."""
        with self._lock:
            frame = self._latest_frame

        if frame is None:
            rospy.loginfo_throttle(10, "No image received yet – skipping API call.")
            return

        # Spin off so we don't block the ROS event loop
        t = threading.Thread(target=self._query_claude, args=(frame,), daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Claude API
    # ------------------------------------------------------------------

    def _frame_to_base64_jpeg(self, frame):
        """Encode a BGR OpenCV frame as a base64 JPEG string."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        success, buf = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            raise RuntimeError("cv2.imencode failed")
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def _query_claude(self, frame):
        """Send image + static prompt to Claude; publish the response."""
        try:
            b64_image = self._frame_to_base64_jpeg(frame)

            response = self._client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": STATIC_PROMPT,
                            },
                        ],
                    }
                ],
            )

            # Extract text from the first content block
            reply = ""
            for block in response.content:
                if block.type == "text":
                    reply = block.text
                    break

            rospy.loginfo("Claude says: %s", reply)
            self._response_pub.publish(String(data=reply))

        except anthropic.APIStatusError as exc:
            rospy.logerr("Anthropic API error %s: %s", exc.status_code, exc.message)
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
