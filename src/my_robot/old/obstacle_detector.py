import rospy
from sensor_msgs.msg import LaserScan

def scan_callback(msg):
    front_distance = msg.ranges[0]
    if front_distance < 1.0:
        print(f"Obstacle ahead! Distance: {front_distance}m")

rospy.init_node('obstacle_detector')
rospy.Subscriber('/scan', LaserScan, scan_callback)
rospy.spin()
