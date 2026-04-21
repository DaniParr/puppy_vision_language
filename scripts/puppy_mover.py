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
        depth_x = msg.x   # Forward depth distance to target (0 if not moving forward)
        depth_y = msg.y   # Lateral depth distance to target (0 if no lateral offset)
        radian_z = msg.z  # Desired final yaw offset (0 if no yaw correction needed)
        self.move_to_target(depth_x, depth_y, radian_z)

    def pose_callback(self, data):
        self.robot_x = data.pose.position.x
        self.robot_y = data.pose.position.y
        orientation_list = [
            data.pose.orientation.x, data.pose.orientation.y,
            data.pose.orientation.z, data.pose.orientation.w
        ]
        (_, _, yaw) = euler_from_quaternion(orientation_list)
        self.robot_yaw = yaw

    def move_to_target(self, depth_x, depth_y, radian_z):
        """
        Three behaviours based on inputs:

        1. Z only (x=0, y=0, z!=0):
           Pure yaw rotation — robot stays in place and rotates by radian_z.

        2. Y only (x=0, y!=0):
           Rotate in place to face the direction of atan2(y, 0+epsilon) — i.e. purely
           sideways. The robot turns to face the target but does not drive forward.
           If Z is also provided, that becomes the final resting yaw after turning.

        3. X (and optional Y) provided (x!=0):
           - Compute the bearing to the target using atan2(y, x).
           - Drive forward sqrt(x²+y²) - STOP_DISTANCE along that bearing.
           - If Z is also provided, rotate to that final yaw once the destination
             is reached. Otherwise hold the approach heading as the final yaw.
        """

        # --- CASE 1: Pure yaw rotation (x=0, y=0, z!=0) ---
        if depth_x == 0.0 and depth_y == 0.0:
            if radian_z == 0.0:
                rospy.logwarn("Received empty goal (x=0, y=0, z=0). Ignoring.")
                return

            # Stay in place; rotate by radian_z relative to current heading
            self.goal_x = self.robot_x
            self.goal_y = self.robot_y
            self.goal_yaw = self.normalize_angle(self.robot_yaw + radian_z)

            rospy.loginfo(
                "CASE 1 — Pure yaw: rotate %.2f rad → target yaw %.2f rad (%.1f°)",
                radian_z, self.goal_yaw, math.degrees(self.goal_yaw)
            )
            self.has_goal = True
            return

        # --- CASE 2: Rotate in place to face target (x=0, y!=0) ---
        if depth_x == 0.0 and depth_y != 0.0:
            # Use atan2(y, x) with x≈0 to get the bearing toward the target.
            bearing_to_target = math.atan2(depth_y, 0.001)  # ±90° depending on y sign

            # Stay in place; rotate to face the bearing
            self.goal_x = self.robot_x
            self.goal_y = self.robot_y

            # If a final yaw is provided, use that; otherwise face the target bearing
            if radian_z != 0.0:
                self.goal_yaw = self.normalize_angle(self.robot_yaw + radian_z)
            else:
                self.goal_yaw = self.normalize_angle(self.robot_yaw + bearing_to_target)

            rospy.loginfo(
                "CASE 2 — Turn to face: bearing %.2f rad, goal yaw %.2f rad (%.1f°)",
                bearing_to_target, self.goal_yaw, math.degrees(self.goal_yaw)
            )
            self.has_goal = True
            return

        # --- CASE 3: Drive toward target (x!=0, y optional) ---
        # Compute bearing and total distance to target using atan2(y, x)
        bearing_to_target = math.atan2(depth_y, depth_x)
        total_distance = math.sqrt(depth_x**2 + depth_y**2)
        travel_distance = total_distance - self.STOP_DISTANCE

        # The heading we need to face to move toward the target
        approach_yaw = self.normalize_angle(self.robot_yaw + bearing_to_target)

        # Final resting yaw: use Z if provided, otherwise hold approach heading
        if radian_z != 0.0:
            self.goal_yaw = self.normalize_angle(self.robot_yaw + radian_z)
        else:
            self.goal_yaw = approach_yaw

        if travel_distance <= self.DIST_TOLERANCE:
            rospy.loginfo("CASE 3 — Already within stop distance. Rotating to goal yaw only.")
            self.goal_x = self.robot_x
            self.goal_y = self.robot_y
            self.has_goal = True
            return

        # Project the travel vector into the global map frame
        # bearing_to_target is relative to the robot, so rotate by robot_yaw
        global_bearing = self.normalize_angle(self.robot_yaw + bearing_to_target)
        self.goal_x = self.robot_x + travel_distance * math.cos(global_bearing)
        self.goal_y = self.robot_y + travel_distance * math.sin(global_bearing)

        rospy.loginfo(
            "CASE 3 — Drive: bearing %.2f rad, dist %.2f m → goal (%.2f, %.2f), final yaw %.2f rad (%.1f°)",
            bearing_to_target, travel_distance,
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