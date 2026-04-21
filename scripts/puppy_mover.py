#!/usr/bin/env python
import math
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Point
from tf.transformations import euler_from_quaternion

class PuppyPiDirectDriver:
    def __init__(self):
        rospy.init_node('puppypi_direct_driver', anonymous=True)
        
        # --- CONFIGURATION & CONSTANTS ---
        self.STOP_DISTANCE = 1.0
        self.RATE = rospy.Rate(10)
        
        # Velocity Limits
        self.MAX_LINEAR_SPEED = 0.3
        self.MAX_ANGULAR_SPEED = 0.5
        
        # Proportional Control Gains
        self.K_LINEAR = 0.5
        self.K_ANGULAR = 1.0
        
        # Tolerances
        self.DIST_TOLERANCE = 0.05   # meters
        self.YAW_TOLERANCE = 0.05    # radians

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
        self.pose_sub = rospy.Subscriber('/slam_out_pose', PoseStamped, self.pose_callback)
        self.target_sub = rospy.Subscriber('/puppy_move', Point, self.target_callback)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        rospy.loginfo("Direct Driver Initialized. Waiting for targets...")

    def target_callback(self, msg):
        depth_x = msg.x
        angle_y = msg.y
        self.move_to_target(depth_x, angle_y)

    def pose_callback(self, data):
        self.robot_x = data.pose.position.x
        self.robot_y = data.pose.position.y
        orientation_list = [
            data.pose.orientation.x, data.pose.orientation.y,
            data.pose.orientation.z, data.pose.orientation.w
        ]
        (_, _, yaw) = euler_from_quaternion(orientation_list)
        self.robot_yaw = yaw

    def move_to_target(self, depth_x, angle_y):
        """Calculates the global stopping point and target yaw, then activates the control loop."""
        travel_distance = depth_x - self.STOP_DISTANCE

        raw_yaw = self.robot_yaw + angle_y
        self.goal_yaw = math.atan2(math.sin(raw_yaw), math.cos(raw_yaw))

        if travel_distance <= self.DIST_TOLERANCE:
            rospy.loginfo("Already at target distance. Will rotate to face target yaw only.")
            self.goal_x = self.robot_x
            self.goal_y = self.robot_y
            self.has_goal = True
            return

        # Calculate relative X/Y displacement
        rel_x = travel_distance * math.cos(angle_y)
        rel_y = travel_distance * math.sin(angle_y)

        # Convert to global map coordinates
        self.goal_x = self.robot_x + (rel_x * math.cos(self.robot_yaw) - rel_y * math.sin(self.robot_yaw))
        self.goal_y = self.robot_y + (rel_x * math.sin(self.robot_yaw) + rel_y * math.cos(self.robot_yaw))

        rospy.loginfo(
            "New Goal Set - X:%.2f, Y:%.2f, Yaw:%.2f rad (%.1f deg)",
            self.goal_x, self.goal_y,
            self.goal_yaw, math.degrees(self.goal_yaw)
        )
        self.has_goal = True

    def normalize_angle(self, angle):
        """Keeps angles within the -pi to pi range."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def control_loop(self):
        """Main loop that drives the robot to the goal in three phases."""
        while not rospy.is_shutdown():
            if not self.has_goal:
                self.RATE.sleep()
                continue

            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            distance_error = math.sqrt(dx**2 + dy**2)

            angle_to_goal = math.atan2(dy, dx)
            heading_error = self.normalize_angle(angle_to_goal - self.robot_yaw)

            # When far away, steer toward the goal point.
            # As distance shrinks, smoothly blend steering toward the final goal_yaw.
            # blend = 0.0 → pure point-tracking; blend = 1.0 → pure goal_yaw tracking
            BLEND_START_DIST = 0.5  # meters at which blending begins
            blend = 1.0 - min(distance_error / BLEND_START_DIST, 1.0)
            final_yaw_error = self.normalize_angle(self.goal_yaw - self.robot_yaw)
            blended_heading_error = (1.0 - blend) * heading_error + blend * final_yaw_error

            twist = Twist()

            # Phase 1: Not facing the target direction — pivot first
            if distance_error > self.DIST_TOLERANCE and abs(heading_error) > 0.2:
                twist.angular.z = self.K_ANGULAR * blended_heading_error
                twist.angular.z = max(min(twist.angular.z, self.MAX_ANGULAR_SPEED), -self.MAX_ANGULAR_SPEED)

            # Phase 2: Facing the target — walk forward while correcting heading
            elif distance_error > self.DIST_TOLERANCE:
                twist.linear.x = self.K_LINEAR * distance_error
                twist.linear.x = max(min(twist.linear.x, self.MAX_LINEAR_SPEED), -self.MAX_LINEAR_SPEED)

                # --- FIX 3: Use blended heading so yaw correction begins before stopping ---
                twist.angular.z = self.K_ANGULAR * blended_heading_error
                twist.angular.z = max(min(twist.angular.z, self.MAX_ANGULAR_SPEED), -self.MAX_ANGULAR_SPEED)

            # Phase 3: At the destination — rotate to final goal_yaw
            else:
                if abs(final_yaw_error) > self.YAW_TOLERANCE:
                    twist.angular.z = self.K_ANGULAR * final_yaw_error
                    twist.angular.z = max(min(twist.angular.z, self.MAX_ANGULAR_SPEED), -self.MAX_ANGULAR_SPEED)
                else:
                    rospy.loginfo("Goal Reached! Final yaw: %.2f rad", self.robot_yaw)
                    self.has_goal = False
                    twist = Twist()

            self.cmd_pub.publish(twist)
            self.RATE.sleep()

if __name__ == '__main__':
    try:
        driver = PuppyPiDirectDriver()
        driver.control_loop()
    except rospy.ROSInterruptException:
        pass