#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from puppy_control.msg import Velocity # Your custom message

def velocity_callback(puppy_msg):
    twist = Twist()
    # Map the puppy's linear speed to the standard Twist linear x
    twist.linear.x = puppy_msg.x 
    # Map the puppy's yaw rate to the standard Twist angular z
    twist.angular.z = puppy_msg.yaw_rate 
    
    cmd_pub.publish(twist)

if __name__ == '__main__':
    rospy.init_node('puppy_to_gazebo_bridge')
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/puppy_control/velocity', Velocity, velocity_callback)
    rospy.spin()