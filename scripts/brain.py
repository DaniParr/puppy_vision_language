#!/usr/bin/env python3
"""
brain.py

Handles all the following logistic data:
    - Cycling through API keys
    - Making calls Gemini
    - Returning formatted output

Compatible with Python 3.7 on Debian 10 / ROS Noetic.

Dependencies:
  sudo apt-get install python3-opencv
  pip3 install "httpx==0.23.3"

Environment variable:
  export GEMINI_API_KEY_1="AIza..."
"""

import base64
import json
import os
import threading
import time
import random

import cv2
import httpx
import rospy
import numpy as np


# Configuration

MODEL_ID       = "gemini-2.5-flash"
MAX_TOKENS     = 512
JPEG_QUALITY   = 75

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

ACTION_FILES = [
    "2_legs_stand.d6ac",
    "bow.d6ac",
    "boxing2.d6ac",
    "boxing.d6ac",
    "grab.d6a",
    "jump.d6ac",
    "kick_ball_left.d6ac",
    "kick_ball_right.d6ac",
    "lie_down.d6ac",
    "look_down.d6ac",
    "moonwalk.d6ac",
    "nod.d6ac",
    "pee.d6ac",
    "place1.d6a",
    "place.d6a",
    "push-up.d6ac",
    "shake_hands.d6ac",
    "shake_head.d6ac",
    "sit.d6ac",
    "spacewalk.d6ac",
    "stand.d6ac",
    "stand_with_arm.d6a",
    "stretch.d6ac",
    "wave.d6ac",
    "scan",
    "move",
    "rotate",
]

TOOLS = [
    {
        "function_declarations": [
            {
                "name": "robot_action",
                "description": (
                    "Select a sequence of action files to execute in response to a prompt. "
                    "Each action may include special parameters depending on its type: "
                    "'scan' requires bounding box info, 'move' requires displacement in meters, "
                    "'rotate' requires angle in radians."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "description": "Ordered list of action files to execute.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_name": {
                                        "type": "string",
                                        "enum": ACTION_FILES,
                                        "description": "The action file to execute.",
                                    },
                                    "action_type": {
                                        "type": "string",
                                        "enum": ["default", "scan", "move", "rotate"],
                                        "description": (
                                            "Type of action. Use 'scan' for visual search, "
                                            "'move' for translation, 'rotate' for turning, "
                                            "'default' for all other actions."
                                        ),
                                    },

                                    # --- scan parameters ---
                                    "scan_center_x": {
                                        "type": "number",
                                        "description": (
                                            "[scan only] Horizontal center of the object of interest "
                                            "in image pixel coordinates."
                                        ),
                                    },
                                    "scan_center_y": {
                                        "type": "number",
                                        "description": (
                                            "[scan only] Vertical center of the object of interest "
                                            "in image pixel coordinates."
                                        ),
                                    },
                                    "scan_width": {
                                        "type": "number",
                                        "description": (
                                            "[scan only] Width of the bounding box around the "
                                            "object of interest in pixels."
                                        ),
                                    },
                                    "scan_height": {
                                        "type": "number",
                                        "description": (
                                            "[scan only] Height of the bounding box around the "
                                            "object of interest in pixels."
                                        ),
                                    },

                                    # --- move parameters ---
                                    "move_meters": {
                                        "type": "number",
                                        "description": (
                                            "[move only] Distance to move in meters. "
                                            "Positive = forward, negative = backward."
                                        ),
                                    },

                                    # --- rotate parameters ---
                                    "rotate_radians": {
                                        "type": "number",
                                        "description": (
                                            "[rotate only] Angle to rotate in radians. "
                                            "Positive = counter-clockwise, negative = clockwise."
                                        ),
                                    },

                                    "description": {
                                        "type": "string",
                                        "description": "Brief human-readable note about why this action was chosen.",
                                    },
                                },
                                "required": ["file_name", "action_type"],
                            },
                        },
                        "response_summary": {
                            "type": "string",
                            "description": "A brief explanation of why these actions were selected to fulfill the prompt.",
                        },
                    },
                    "required": ["actions", "response_summary"],
                },
            }
        ]
    }
]

class Brain:
    """
    Class that handles making the API calls and interpretting responses.
    """

    def __init__(self):

        # Load known API Keys
        self._api_keys = [
            os.environ.get("GEMINI_API_KEY_0"),
            os.environ.get("GEMINI_API_KEY_1"),
            os.environ.get("GEMINI_API_KEY_2"),
        ]

        # Check if all API keys exist
        if not any(self._api_keys):
            rospy.logfatal("GEMINI_API_KEY_0, GEMINI_API_KEY_1, and GEMINI_API_KEY_2 environment variables not set. Exiting now.")
            raise SystemExit(1)

        # Set up API
        self._api_url = GEMINI_API_URL.format(model=MODEL_ID)
        self._headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": random.choice(self._api_keys),
        }

    def send_request(self, prompt: str, frame: np.array) -> (str, list):
        """
        Sends HTTP request to gemini API.
        """
        try:

            # Ready the payload
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
                            {"text": prompt},
                        ]
                    }
                ],
                "tools": TOOLS,
                "generationConfig": {
                    "maxOutputTokens": MAX_TOKENS,
                    "temperature": 0.2,  # low temp for consistent structured output
                },
            }

            # Send Request
            rospy.loginfo("Sending image to Gemini...")

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    self._api_url,
                    headers=self._headers,
                    content=json.dumps(payload),
                )

            # Set new API key for header
            self._headers["x-goog-api-key"] = random.choice(self._api_keys)

            # Get response
            if resp.status_code != 200:
                rospy.logerr("API error %d: %s", resp.status_code, resp.text[:300])
                return "", []

            data = resp.json()
            return self._handle_response(data)

        except httpx.TimeoutException:
            rospy.logerr("Gemini API request timed out.")
            return "", []
        
    def _encode_frame(self, frame):
        
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        ok, buf = cv2.imencode(".jpg", frame, params)
        
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")
    
    def _handle_response(self, data) -> (str, list):
        """
        Extract the robot_action function call from Gemini's response.
        Returns (summary, actions) or (None, None) on failure.
        """
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as exc:
            rospy.logerr("Unexpected response structure: %s", exc)
            return "", []

        for part in parts:
            if "functionCall" not in part:
                continue

            fn      = part["functionCall"]
            fn_name = fn.get("name")
            fn_args = fn.get("args", {})

            if fn_name != "robot_action":
                rospy.logwarn("Unexpected function call: %s", fn_name)
                continue

            summary = fn_args.get("response_summary", "")
            actions = fn_args.get("actions", [])

            rospy.loginfo("Summary: %s | %d action(s) received.", summary, len(actions))
            return summary, actions

        rospy.logwarn("No valid robot_action function call found in response.")
        return "", []
