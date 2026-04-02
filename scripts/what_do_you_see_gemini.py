#!/usr/bin/env python3
"""
gemini_vision_node.py

ROS1 Noetic node that:
  - Subscribes to /usb_cam/image_raw (sensor_msgs/Image)
  - Converts the latest frame to a base64 JPEG
  - Sends it to Google Gemini Vision API via raw HTTP (no heavy SDK needed)
  - Publishes the model's text response to /gemini_vision/response (std_msgs/String)

Compatible with Python 3.7 on Debian 10 / ROS Noetic.

Dependencies (all Python 3.7 compatible):
  sudo apt-get install python3-opencv
  pip3 install "httpx==0.23.3"

Get a free API key (no credit card required):
  https://aistudio.google.com → "Get API key"

Environment variable (set before running):
  export GEMINI_API_KEY="AIza..."

Usage:
  rosrun <your_package> gemini_vision_node.py
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
MODEL_ID       = "gemini-1.5-flash"   # fast, free-tier, vision-capable
MAX_TOKENS     = 512
JPEG_QUALITY   = 75    # 0-100; lower = smaller payload, faster upload
QUERY_INTERVAL = 5.0   # seconds between API calls

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)


class GeminiVisionNode:
    def __init__(self):
        rospy.init_node("gemini_vision_node", anonymous=False)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            rospy.logfatal("GEMINI_API_KEY environment variable is not set. Exiting.")
            raise SystemExit(1)

        self._api_url = GEMINI_API_URL.format(model=MODEL_ID, key=api_key)

        self._bridge       = CvBridge()
        self._lock         = threading.Lock()
        self._latest_frame = None

        # Publisher
        self._response_pub = rospy.Publisher(
            "/gemini_vision/response", String, queue_size=10
        )

        # Subscriber — queue_size=1 keeps only the freshest frame
        rospy.Subscriber(
            "/usb_cam/image_raw", Image, self._image_callback,
            queue_size=1, buff_size=2 ** 24
        )

        # Timer — throttle API calls regardless of camera frame rate
        rospy.Timer(rospy.Duration(QUERY_INTERVAL), self._timer_callback)

        rospy.loginfo(
            "gemini_vision_node ready. Querying Gemini every %.1f s.", QUERY_INTERVAL
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _image_callback(self, msg):
        """Cache the latest image — no API call here."""
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._lock:
                self._latest_frame = frame
        except Exception as exc:
            rospy.logwarn("cv_bridge conversion failed: %s", exc)

    def _timer_callback(self, _event):
        """Grab cached frame and fire API call in a background thread."""
        with self._lock:
            frame = self._latest_frame

        if frame is None:
            rospy.loginfo_throttle(10, "No image received yet — skipping API call.")
            return

        t = threading.Thread(target=self._query_gemini, args=(frame,), daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Gemini API (raw HTTP — no SDK required)
    # ------------------------------------------------------------------

    def _encode_frame(self, frame):
        """Return a base64-encoded JPEG string for the given BGR frame."""
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", frame, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def _query_gemini(self, frame):
        """Build the multimodal request, POST it, and publish the reply."""
        try:
            b64_image = self._encode_frame(frame)

            # Gemini multimodal request format
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
                            {
                                "text": STATIC_PROMPT
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": MAX_TOKENS,
                    "temperature": 0.4,
                },
            }

            headers = {"Content-Type": "application/json"}

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    self._api_url,
                    headers=headers,
                    content=json.dumps(payload),
                )

            if resp.status_code != 200:
                rospy.logerr(
                    "Gemini API error %d: %s", resp.status_code, resp.text[:300]
                )
                return

            data = resp.json()

            # Extract text from response
            try:
                reply = (
                    data["candidates"][0]["content"]["parts"][0]["text"]
                )
            except (KeyError, IndexError) as exc:
                rospy.logerr("Unexpected response structure: %s | raw: %s", exc, data)
                return

            rospy.loginfo("Gemini: %s", reply)
            self._response_pub.publish(String(data=reply))

        except httpx.TimeoutException:
            rospy.logerr("Request to Gemini API timed out.")
        except Exception as exc:
            rospy.logerr("Unexpected error in _query_gemini: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        node = GeminiVisionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
