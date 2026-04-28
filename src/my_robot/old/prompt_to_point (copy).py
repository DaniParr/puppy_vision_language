#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point
import re

def prompt_callback(msg):
    text = msg.data.lower().strip()
    rospy.loginfo(f"Received prompt: '{text}'")
    
    point = Point()
    
    # forward
    if re.search(r"move forward|go forward|walk forward", text):
        point.x = 1.0
        point.y = 0.0
        point.z = 0.0

    # backward
    elif re.search(r"move backward|go backward|walk backward", text):
        point.x = -1.0
        point.y = 0.0
        point.z = 0.0

    # turn left
    elif re.search(r"turn left|rotate left", text):
        point.x = 0.0
        point.y = 0.0
        point.z = 0.5

    # turn right
    elif re.search(r"turn right|rotate right", text):
        point.x = 0.0
        point.y = 0.0
        point.z = -0.5

    # stop
    elif re.search(r"stop|halt", text):
        point.x = 0.0
        point.y = 0.0
        point.z = 0.0

    else:
        rospy.logwarn(f"Unknown command: '{text}' — ignoring.")
        return

    rospy.loginfo(f"Publishing Point: x={point.x}, y={point.y}, z={point.z}")
    point_pub.publish(point)

if __name__ == '__main__':
    rospy.init_node('prompt_to_point')
    point_pub = rospy.Publisher('/puppy_move', Point, queue_size=10)
    rospy.Subscriber('/prompt', String, prompt_callback)
    rospy.loginfo("Prompt to Point bridge ready!")
    rospy.spin()
