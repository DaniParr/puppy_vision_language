#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point
import re

def prompt_callback(msg):
    text = msg.data.lower().strip()
    rospy.loginfo(f"Received prompt: '{text}'")
    
    point = Point()
    matched = True

    # Move forward X meters
    match = re.search(r"move forward (\d+\.?\d*)", text)
    if match:
        point.x = float(match.group(1))
        point.y = 0.0
        point.z = 0.0

    # Move backward X meters
    elif re.search(r"move backward (\d+\.?\d*)", text):
        match = re.search(r"move backward (\d+\.?\d*)", text)
        point.x = -float(match.group(1))
        point.y = 0.0
        point.z = 0.0

    # Move left X meters
    elif re.search(r"move left (\d+\.?\d*)", text):
        match = re.search(r"move left (\d+\.?\d*)", text)
        point.x = 0.0
        point.y = float(match.group(1))
        point.z = 0.0

    # Move right X meters
    elif re.search(r"move right (\d+\.?\d*)", text):
        match = re.search(r"move right (\d+\.?\d*)", text)
        point.x = 0.0
        point.y = -float(match.group(1))
        point.z = 0.0

    # Turn left X degrees
    elif re.search(r"turn left (\d+\.?\d*)", text):
        match = re.search(r"turn left (\d+\.?\d*)", text)
        import math
        point.x = 0.0
        point.y = 0.0
        point.z = float(match.group(1)) * (math.pi / 180)

    # Turn right X degrees
    elif re.search(r"turn right (\d+\.?\d*)", text):
        match = re.search(r"turn right (\d+\.?\d*)", text)
        import math
        point.x = 0.0
        point.y = 0.0
        point.z = -float(match.group(1)) * (math.pi / 180)

    # Move to coordinates: "go to 2 3" → x=2, y=3
    elif re.search(r"go to (-?\d+\.?\d*) (-?\d+\.?\d*)", text):
        match = re.search(r"go to (-?\d+\.?\d*) (-?\d+\.?\d*)", text)
        GRID_SCALE = 0.1  # tune this to match one grid square
        point.x = float(match.group(1)) * GRID_SCALE
        point.y = float(match.group(2)) * GRID_SCALE
        point.z = 0.0

    # Stop
    elif re.search(r"stop|halt", text):
        point.x = 0.0
        point.y = 0.0
        point.z = 0.0

    else:
        rospy.logwarn(f"Unknown command: '{text}'")
        matched = False

    if matched:
        rospy.loginfo(f"Publishing target: x={point.x}, y={point.y}, z={point.z}")
        point_pub.publish(point)

if __name__ == '__main__':
    rospy.init_node('prompt_to_point')
    point_pub = rospy.Publisher('/puppy_move', Point, queue_size=10)
    rospy.Subscriber('/prompt', String, prompt_callback)
    rospy.loginfo("Prompt to Point ready!")
    rospy.spin()
