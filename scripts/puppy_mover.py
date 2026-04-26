#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import PoseStamped, Twist, Point
from tf.transformations import euler_from_quaternion
from puppy_control.msg import Velocity, Pose


class PuppyPiDirectDriver:
    def __init__(self):
        rospy.init_node('puppypi_direct_driver', anonymous=True)
        
        # --- CONFIGURATION & CONSTANTS ---
        self.STOP_DISTANCE = 0.05    # meters — goal point placed well short of target
        self.RATE = rospy.Rate(10)
        self.pose_received = False

        # Velocity Limits (raw hardware units)
        self.MAX_LINEAR_SPEED = 15
        self.MAX_ANGULAR_SPEED = 0.5
        
        self.MIN_LINEAR_SPEED = 7
        self.MIN_ANGULAR_SPEED = 0.2

        # Proportional Control Gains
        self.K_LINEAR = 50.0
        self.K_ANGULAR = 1.0
        
        # Tolerances
        self.DIST_TOLERANCE = 0.05 
        self.YAW_TOLERANCE = 0.15    

        # --- STATE VARIABLES ---
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        STAND = Pose()
        STAND.roll = math.radians(0)
        STAND.pitch = math.radians(0)
        STAND.yaw = 0.000
        STAND.height = -10
        STAND.x_shift = 0.4
        STAND.stance_x = 0
        STAND.stance_y = 0
        STAND.run_time = 2

        # Goal State
        self.has_goal = False
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.has_explicit_yaw = False  # True only when radian_z was provided
        
        # Latch to prevent tolerance chattering between phases
        self.position_reached = False 
        
        # Latch to prevent dynamically flipping gears and spiraling
        self.is_forward_move = True

        # --- ROS INTERFACES ---
        self.pose_sub = rospy.Subscriber('/slam_out_pose', PoseStamped, self.pose_callback)
        self.target_sub = rospy.Subscriber('/puppy_move', Point, self.target_callback)
        self.pup_velocity_pub = rospy.Publisher('/puppy_control/velocity', Velocity, queue_size=10)
        self.pup_pose_pub = rospy.Publisher('/puppy_control/pose', Pose, queue_size=10)
        self.pup_pose_pub.publish(STAND)

        rospy.on_shutdown(self.stop_robot)
        rospy.loginfo("Direct Driver Initialized. Waiting for targets...")

    def target_callback(self, msg):
        depth_x = msg.x
        depth_y = msg.y
        radian_z = msg.z
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

    def apply_velocity_limits(self, velocity, max_speed, min_speed):
        if velocity == 0.0:
            return 0.0
        clamped = max(min(velocity, max_speed), -max_speed)
        if abs(clamped) < min_speed:
            return math.copysign(min_speed, clamped)
        return clamped

    def move_to_target(self, depth_x, depth_y, radian_z):
        # --- CASE 1: Pure yaw rotation (x=0, y=0, z!=0) ---
        if depth_x == 0.0 and depth_y == 0.0:
            if radian_z == 0.0:
                rospy.logwarn("Received empty goal (x=0, y=0, z=0). Ignoring.")
                return

            self.goal_x = self.robot_x
            self.goal_y = self.robot_y
            self.goal_yaw = self.normalize_angle(self.robot_yaw + radian_z)

            rospy.loginfo(
                "CASE 1 — Pure yaw: rotate %.2f rad → target yaw %.2f rad (%.1f°)",
                radian_z, self.goal_yaw, math.degrees(self.goal_yaw)
            )
            self.has_explicit_yaw = True
            self.has_goal = True
            self.position_reached = False
            self.is_forward_move = True
            return

        # --- CASE 2: Rotate in place to face target (x=0, y!=0) ---
        if depth_x == 0.0 and depth_y != 0.0:
            bearing_to_target = math.atan2(depth_y, 0.001)

            self.goal_x = self.robot_x
            self.goal_y = self.robot_y

            if radian_z != 0.0:
                self.goal_yaw = self.normalize_angle(self.robot_yaw + radian_z)
            else:
                self.goal_yaw = self.normalize_angle(self.robot_yaw + bearing_to_target)

            rospy.loginfo(
                "CASE 2 — Turn to face: bearing %.2f rad, goal yaw %.2f rad (%.1f°)",
                bearing_to_target, self.goal_yaw, math.degrees(self.goal_yaw)
            )
            self.has_explicit_yaw = True  
            self.has_goal = True
            self.position_reached = False
            self.is_forward_move = True
            return

        # --- CASE 3: Drive toward target (x!=0, y optional) ---

        # --- CASE 3R: Reverse ---
        if depth_x < 0.0:
            total_distance = abs(depth_x)
            stop_dist = min(self.STOP_DISTANCE, total_distance * 0.2)
            travel_distance = total_distance - stop_dist
            self.goal_yaw = self.robot_yaw  
            if radian_z != 0.0:
                self.goal_yaw = self.normalize_angle(self.robot_yaw + radian_z)
            self.has_explicit_yaw = (radian_z != 0.0)
            reverse_bearing = self.normalize_angle(self.robot_yaw + math.pi)
            self.goal_x = self.robot_x + travel_distance * math.cos(reverse_bearing)
            self.goal_y = self.robot_y + travel_distance * math.sin(reverse_bearing)
            rospy.loginfo(
                "CASE 3R — Reverse: dist %.2f m → goal (%.2f, %.2f)",
                travel_distance, self.goal_x, self.goal_y
            )
            self.has_goal = True
            self.position_reached = False
            self.is_forward_move = False # Lock reverse gear
            return

        bearing_to_target = math.atan2(depth_y, depth_x)
        total_distance = math.sqrt(depth_x**2 + depth_y**2)
        stop_dist = min(self.STOP_DISTANCE, total_distance * 0.2)
        travel_distance = total_distance - stop_dist

        approach_yaw = self.normalize_angle(self.robot_yaw + bearing_to_target)

        if radian_z != 0.0:
            self.goal_yaw = self.normalize_angle(self.robot_yaw + radian_z)
        else:
            self.goal_yaw = approach_yaw
        self.has_explicit_yaw = (radian_z != 0.0)

        if travel_distance <= self.DIST_TOLERANCE:
            rospy.loginfo("CASE 3 — Already within stop distance. Rotating to goal yaw only.")
            self.goal_x = self.robot_x
            self.goal_y = self.robot_y
            self.has_goal = True
            self.position_reached = False
            self.is_forward_move = True
            return

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
        self.position_reached = False
        self.is_forward_move = True # Lock forward gear

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def publish_velocity(self, velocity):
        self.pup_velocity_pub.publish(velocity)

    def stop_robot(self):
        self.publish_velocity(Velocity())

    def control_loop(self):
        """Main loop that drives the robot to the goal in three phases."""
        
        # Distance at which we stop steering toward the specific X/Y point 
        # and just lock our heading to coast in smoothly.
        COAST_DISTANCE = 0.20 
        
        while not rospy.is_shutdown():
            if not self.has_goal:
                self.RATE.sleep()
                continue

            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            distance_error = math.sqrt(dx**2 + dy**2)

            angle_to_goal = math.atan2(dy, dx)
            heading_error = self.normalize_angle(angle_to_goal - self.robot_yaw)
            final_yaw_error = self.normalize_angle(self.goal_yaw - self.robot_yaw)

            if self.is_forward_move:
                driving_heading_error = heading_error
            else:
                driving_heading_error = self.normalize_angle(heading_error + math.pi)

            approach_heading_error = driving_heading_error

            rospy.loginfo_throttle(
                0.5,
                "dist_err=%.3fm heading_err=%.2frad robot=(%.2f,%.2f) goal=(%.2f,%.2f)",
                distance_error, heading_error,
                self.robot_x, self.robot_y,
                self.goal_x, self.goal_y
            )

            velocity = Velocity()

            # --- THE HARD LATCH ---
            # If we cross the 5cm finish line, lock the state to True.
            if distance_error <= self.DIST_TOLERANCE:
                self.position_reached = True

            # If we haven't reached the goal yet, do Phase 1 or 2
            if not self.position_reached:
                
                # Phase 1: Severely off-course — pure pivot, no forward motion.
                # Only allowed if we are far away (> COAST_DISTANCE) to prevent 
                # violent spinning when we are practically right next to the goal.
                if distance_error > COAST_DISTANCE and abs(driving_heading_error) > math.pi / 2 and self.goal_x != self.robot_x:
                    velocity.yaw_rate = self.K_ANGULAR * approach_heading_error
                    velocity.yaw_rate = self.apply_velocity_limits(
                        velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                    )

                # Phase 2: Walk toward goal.
                else:
                    linear_speed = self.K_LINEAR * distance_error
                    linear_speed = self.apply_velocity_limits(
                        linear_speed, self.MAX_LINEAR_SPEED, self.MIN_LINEAR_SPEED
                    )
                    
                    if self.is_forward_move:
                        velocity.x = linear_speed
                    else:
                        velocity.x = -linear_speed

                    # Proximity Coast: If we are close, stop tracking the violent X/Y angle 
                    # and just lock onto the final resting yaw to coast straight in.
                    if distance_error < COAST_DISTANCE:
                        velocity.yaw_rate = self.K_ANGULAR * final_yaw_error
                    else:
                        velocity.yaw_rate = self.K_ANGULAR * approach_heading_error

                    velocity.yaw_rate = self.apply_velocity_limits(
                        velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                    )

            # Phase 3: At the destination — rotate to correct any yaw drift.
            # Because of the Hard Latch above, even if the robot drifts backward 10cm 
            # while turning, it will stay stuck in this loop until the turn is done!
            else:
                if abs(final_yaw_error) > self.YAW_TOLERANCE:
                    velocity.yaw_rate = self.K_ANGULAR * final_yaw_error
                    velocity.yaw_rate = self.apply_velocity_limits(
                        velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                    )
                else:
                    rospy.loginfo("Goal Reached! Final yaw: %.2f rad", self.robot_yaw)
                    self.has_goal = False
                    self.position_reached = False # Reset latch for the next command
                    self.stop_robot()
                    continue

            self.publish_velocity(velocity)
            self.RATE.sleep()


if __name__ == '__main__':
    try:
        driver = PuppyPiDirectDriver()
        driver.control_loop()
    except rospy.ROSInterruptException:
        pass