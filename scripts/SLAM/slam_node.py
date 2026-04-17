#!/usr/bin/env python
import math
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Point
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Int32

class PuppyPiLLMInterface:
    def __init__(self):
        rospy.init_node('puppypi_llm_interface', anonymous=True)
        
        # Camera Constants
        self.IMAGE_WIDTH = 640.0 
        self.FOV_H = math.radians(120.0) 

        # State Variables
        self.latest_scan = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0 # Heading in radians
        
        # Subscribers
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        # Subscribe to Hector SLAM's output
        self.pose_sub = rospy.Subscriber('/slam_out_pose', PoseStamped, self.pose_callback)
        self.pixel_sub = rospy.Subscriber('/ai_target_pixel', Int32, self.pixel_callback)

        
        # Publisher for LLM to read/use
        # Publishing a simple Point (X, Y, Z) representing the target destination
        self.target_pub = rospy.Publisher('/llm_navigation_goal', Point, queue_size=10)

        rospy.loginfo("PuppyPi LLM Interface Node Initialized.")

    def scan_callback(self, data):
        self.latest_scan = data

    def pose_callback(self, data):
        # Hector SLAM updates the robot's location here
        self.robot_x = data.pose.position.x
        self.robot_y = data.pose.position.y
        
        # Convert quaternion orientation to Euler angles to get the Yaw (heading)
        orientation_q = data.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.robot_yaw = yaw

    def calculate_global_target(self, pixel_x):
        """
        Translates a 2D camera pixel into a Global Map Coordinate.
        """
        if self.latest_scan is None:
            rospy.logwarn("No LiDAR data yet.")
            return None

        # 1. Get angle from camera center
        center_x = self.IMAGE_WIDTH / 2.0
        target_angle = ((center_x - pixel_x) / self.IMAGE_WIDTH) * self.FOV_H

        try:
            # 2. Get distance from LiDAR
            index = int((target_angle - self.latest_scan.angle_min) / self.latest_scan.angle_increment)
            distance = self.latest_scan.ranges[index]

            if math.isinf(distance) or math.isnan(distance):
                return None

            # 3. Calculate Relative X/Y (Distance directly in front and to the side of the robot)
            rel_x = distance * math.cos(target_angle)
            rel_y = distance * math.sin(target_angle)

            # 4. Convert to Global Map Coordinates using the Hector SLAM Pose
            global_x = self.robot_x + (rel_x * math.cos(self.robot_yaw) - rel_y * math.sin(self.robot_yaw))
            global_y = self.robot_y + (rel_x * math.sin(self.robot_yaw) + rel_y * math.cos(self.robot_yaw))

            return global_x, global_y

        except Exception as e:
            rospy.logerr("Calculation error: %s", str(e))
            return None

    def publish_target_for_llm(self, pixel_x):
        # This is the function the script will call when the LLM finds a target
        coords = self.calculate_global_target(pixel_x)
        
        if coords:
            global_x, global_y = coords
            
            # Create a point message and publish it
            target_msg = Point()
            target_msg.x = global_x
            target_msg.y = global_y
            target_msg.z = 0.0 # 2D map, so Z is flat
            
            self.target_pub.publish(target_msg)
            rospy.loginfo("Published new LLM target to /llm_navigation_goal: X:%.2f, Y:%.2f", global_x, global_y)

    def pixel_callback(self, msg):
        pixel_x = msg.data
        rospy.loginfo("Received target at pixel %d from LLM. Processing...", pixel_x)
        
        self.publish_target_for_llm(pixel_x)


    def run(self):
        rospy.spin() # Keeps the node alive listening to Hector and LiDAR

if __name__ == '__main__':
    try:
        interface = PuppyPiLLMInterface()
        
        # interface.publish_target_for_llm(320) # Example: Simulate a target at the center pixel (320) for testing
        
        interface.run()
    except rospy.ROSInterruptException:
        pass