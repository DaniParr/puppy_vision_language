#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('prompt', String, queue_size=10)
    rospy.init_node('talker_node', anonymous=True)
    
    while not rospy.is_shutdown():
        # Keep the topic alive so we can publish to it
        pass

if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
