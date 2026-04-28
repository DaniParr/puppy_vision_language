#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from puppy_control.msg import Velocity

SPEED_SCALE = 0.05

def velocity_callback(puppy_msg):
    twist = Twist()
    twist.linear.x = puppy_msg.x * SPEED_SCALE
    twist.angular.z = puppy_msg.yaw_rate * SPEED_SCALE
    cmd_pub.publish(twist)

def odom_callback(msg):
    # Convert Gazebo odometry to slam_out_pose so puppy_mover knows where it is
    pose = PoseStamped()
    pose.header = msg.header
    pose.pose = msg.pose.pose
    slam_pub.publish(pose)

if __name__ == '__main__':
    rospy.init_node('puppy_to_gazebo_bridge')
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    slam_pub = rospy.Publisher('/slam_out_pose', PoseStamped, queue_size=10)
    rospy.Subscriber('/puppy_control/velocity', Velocity, velocity_callback)
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rospy.spin()
