#!/usr/bin/env python
import math
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Point
from tf.transformations import euler_from_quaternion

class PuppyPiDirectDriver:
    def __init__(self):
        rospy.init_node('puppypi_direct_driver', anonymous=True)
        
        # --- CONFIGURATION & CONSTANTS ---
        self.STOP_DISTANCE = 1.0    # Stop 1 meter away
        self.RATE = rospy.Rate(10)  # Run control loop at 10 Hz
        
        # Velocity Limits
        self.MAX_LINEAR_SPEED = 0.3  # m/s
        self.MAX_ANGULAR_SPEED = 0.5 # rad/s
        
        # Proportional Control Gains
        self.K_LINEAR = 0.5   # Aggressiveness of forward movement
        self.K_ANGULAR = 1.0  # Aggressiveness of turning
        
        # Tolerances
        self.DIST_TOLERANCE = 0.05  # meters
        self.YAW_TOLERANCE = 0.05   # radians

        # --- STATE VARIABLES ---
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0 
        
        # Goal State
        self.has_goal = False
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0

        # --- ROS INTERFACES ---
        # Track location via Hector SLAM
        self.pose_sub = rospy.Subscriber('/slam_out_pose', PoseStamped, self.pose_callback)
        self.target_sub = rospy.Subscriber('/puppy_move', Point, self.target_callback)

        # Publish direct movement commands
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        

        rospy.loginfo("Direct Driver Initialized. Waiting for targets...")

    def target_callback(self, msg):
        depth_x = msg.x
        angle_y = msg.y
        self.move_to_target(depth_x, angle_y)


    def pose_callback(self, data):
        self.robot_x = data.pose.position.x
        self.robot_y = data.pose.position.y
        orientation_list = [data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.robot_yaw = yaw

    def move_to_target(self, depth_x, angle_y):
        """ Calculates the global stopping point and activates the control loop """
        travel_distance = depth_x - self.STOP_DISTANCE

        if travel_distance <= self.DIST_TOLERANCE:
            rospy.loginfo("Already at target distance. Stopping.")
            self.stop_robot()
            self.has_goal = False
            return

        # Calculate Relative X/Y based on the travel distance
        rel_x = travel_distance * math.cos(angle_y)
        rel_y = travel_distance * math.sin(angle_y)

        # Convert to Global Map Coordinates using Hector SLAM Pose
        self.goal_x = self.robot_x + (rel_x * math.cos(self.robot_yaw) - rel_y * math.sin(self.robot_yaw))
        self.goal_y = self.robot_y + (rel_x * math.sin(self.robot_yaw) + rel_y * math.cos(self.robot_yaw))

        # Calculate target yaw
        raw_yaw = self.robot_yaw + angle_y
        self.goal_yaw = math.atan2(math.sin(raw_yaw), math.cos(raw_yaw))

        rospy.loginfo("New Goal Set - X:%.2f, Y:%.2f", self.goal_x, self.goal_y)
        self.has_goal = True

    def normalize_angle(self, angle):
        """ Keeps angles within the -pi to pi range """
        return math.atan2(math.sin(angle), math.cos(angle))

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    def control_loop(self):
        """ The main loop that drives the robot to the goal """
        while not rospy.is_shutdown():
            if not self.has_goal:
                self.RATE.sleep()
                continue

            # 1. Calculate errors
            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            distance_error = math.sqrt(dx**2 + dy**2)
            
            angle_to_goal = math.atan2(dy, dx)
            heading_error = self.normalize_angle(angle_to_goal - self.robot_yaw)

            twist = Twist()

            # 2. Phase 1: If we are not facing the target point, pivot first
            if abs(heading_error) > 0.2: # Roughly 11 degrees
                twist.angular.z = self.K_ANGULAR * heading_error
                # Clamp angular speed
                twist.angular.z = max(min(twist.angular.z, self.MAX_ANGULAR_SPEED), -self.MAX_ANGULAR_SPEED)
                
            # 3. Phase 2: We are facing the point, now walk to it
            elif distance_error > self.DIST_TOLERANCE:
                twist.linear.x = self.K_LINEAR * distance_error
                # Clamp linear speed
                twist.linear.x = max(min(twist.linear.x, self.MAX_LINEAR_SPEED), -self.MAX_LINEAR_SPEED)
                
                # Keep correcting heading slightly while walking
                twist.angular.z = self.K_ANGULAR * heading_error
                twist.angular.z = max(min(twist.angular.z, self.MAX_ANGULAR_SPEED), -self.MAX_ANGULAR_SPEED)

            # 4. Phase 3: We reached the point, do a final rotation to face the exact target yaw
            else:
                final_yaw_error = self.normalize_angle(self.goal_yaw - self.robot_yaw)
                if abs(final_yaw_error) > self.YAW_TOLERANCE:
                    twist.angular.z = self.K_ANGULAR * final_yaw_error
                    twist.angular.z = max(min(twist.angular.z, self.MAX_ANGULAR_SPEED), -self.MAX_ANGULAR_SPEED)
                else:
                    rospy.loginfo("Goal Reached!")
                    self.has_goal = False
                    twist = Twist()

            self.cmd_pub.publish(twist)
            self.RATE.sleep()

if __name__ == '__main__':
    try:
        driver = PuppyPiDirectDriver()
        
        # In a real scenario, you would call this when you detect a target:
        # Start the continuous control loop
        driver.control_loop()
    except rospy.ROSInterruptException:
        pass