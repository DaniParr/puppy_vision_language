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
        
        self.MIN_LINEAR_SPEED = 3
        self.MIN_ANGULAR_SPEED = 0.2

        # Proportional Control Gains
        self.K_LINEAR = 50.0
        self.K_ANGULAR = 1.0
        
        # Tolerances
        self.DIST_TOLERANCE = 0.05 
        self.YAW_TOLERANCE = 0.3    # radians

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
        """Clamps to max, and if non-zero, enforces a minimum so the hardware actually moves."""
        if velocity == 0.0:
            return 0.0
        clamped = max(min(velocity, max_speed), -max_speed)
        if abs(clamped) < min_speed:
            return math.copysign(min_speed, clamped)
        return clamped

    def move_to_target(self, depth_x, depth_y, radian_z):
        """
        Three behaviours based on inputs:

        1. Z only (x=0, y=0, z!=0):
           Pure yaw rotation — robot stays in place and rotates by radian_z.

        2. Y only (x=0, y!=0):
           Rotate in place to face the direction of atan2(y, 0+epsilon).
           If Z is also provided, that becomes the final resting yaw.

        3. X (and optional Y) provided (x!=0):
           Drive forward sqrt(x²+y²) - stop_dist along the bearing atan2(y, x),
           where stop_dist = min(STOP_DISTANCE, total_distance * 0.2) so that
           short commands still have meaningful travel distance.
           If Z is also provided, rotate to that final yaw once the destination
           is reached. Otherwise hold the approach heading as the final yaw.
        """

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
            self.has_explicit_yaw = True  # Case 2 always rotates to a specific yaw
            self.has_goal = True
            self.position_reached = False
            return

        # --- CASE 3: Drive toward target (x!=0, y optional) ---

        # --- CASE 3R: Reverse — negative x means drive straight backward ---
        # Skip bearing/pivot logic entirely; project the goal directly behind
        # the robot so it walks backward without turning around.
        if depth_x < 0.0:
            total_distance = abs(depth_x)
            stop_dist = min(self.STOP_DISTANCE, total_distance * 0.2)
            travel_distance = total_distance - stop_dist
            self.goal_yaw = self.robot_yaw  # hold current heading throughout
            if radian_z != 0.0:
                self.goal_yaw = self.normalize_angle(self.robot_yaw + radian_z)
            self.has_explicit_yaw = (radian_z != 0.0)
            # Project goal directly behind the robot in the global frame
            reverse_bearing = self.normalize_angle(self.robot_yaw + math.pi)
            self.goal_x = self.robot_x + travel_distance * math.cos(reverse_bearing)
            self.goal_y = self.robot_y + travel_distance * math.sin(reverse_bearing)
            rospy.loginfo(
                "CASE 3R — Reverse: dist %.2f m → goal (%.2f, %.2f)",
                travel_distance, self.goal_x, self.goal_y
            )
            self.has_goal = True
            self.position_reached = False
            return

        bearing_to_target = math.atan2(depth_y, depth_x)
        total_distance = math.sqrt(depth_x**2 + depth_y**2)
        # Scale stop buffer to 20% of total distance, capped at STOP_DISTANCE.
        # This prevents short commands (e.g. 0.3m) from having their entire
        # travel distance consumed by a fixed stop buffer.
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

    def normalize_angle(self, angle):
        """Keeps angles within the -pi to pi range."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def publish_velocity(self, velocity):
        """Single publish boundary for all velocity commands."""
        self.pup_velocity_pub.publish(velocity)

    def stop_robot(self):
        self.publish_velocity(Velocity())

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
            final_yaw_error = self.normalize_angle(self.goal_yaw - self.robot_yaw)

            # REMOVED BLEND FUNCTION: 
            # Always point directly at the goal coordinates until we actually arrive.
            approach_heading_error = heading_error

            rospy.loginfo_throttle(
                0.5,
                "dist_err=%.3fm heading_err=%.2frad robot=(%.2f,%.2f) goal=(%.2f,%.2f)",
                distance_error, heading_error,
                self.robot_x, self.robot_y,
                self.goal_x, self.goal_y
            )

            velocity = Velocity()

            # HYSTERESIS LOGIC:
            # If we already made it to the goal, double the allowed tolerance. 
            # This prevents the physical drift of turning in place from throwing 
            # the robot back into Phase 2 (driving forward).
            active_tolerance = self.DIST_TOLERANCE * 2.0 if self.position_reached else self.DIST_TOLERANCE

            # Phase 1: Severely off-course — pure pivot, no forward motion.
            if distance_error > active_tolerance and abs(heading_error) > math.pi / 2 and self.goal_x != self.robot_x:
                self.position_reached = False
                velocity.yaw_rate = self.K_ANGULAR * approach_heading_error
                velocity.yaw_rate = self.apply_velocity_limits(
                    velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                )

            # Phase 2: Walk toward goal while correcting heading simultaneously.
            elif distance_error > active_tolerance:
                self.position_reached = False
                linear_speed = self.K_LINEAR * distance_error
                linear_speed = self.apply_velocity_limits(
                    linear_speed, self.MAX_LINEAR_SPEED, self.MIN_LINEAR_SPEED
                )
                
                # If goal is roughly behind us, drive backward
                if abs(heading_error) > math.pi / 2:
                    velocity.x = -linear_speed
                else:
                    velocity.x = linear_speed

                velocity.yaw_rate = self.K_ANGULAR * approach_heading_error
                velocity.yaw_rate = self.apply_velocity_limits(
                    velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                )

            # Phase 3: At the destination — rotate to correct any yaw drift.
            else:
                self.position_reached = True # Latch the state! We made it to the XY coords.
                
                if abs(final_yaw_error) > self.YAW_TOLERANCE:
                    velocity.yaw_rate = self.K_ANGULAR * final_yaw_error
                    velocity.yaw_rate = self.apply_velocity_limits(
                        velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                    )
                else:
                    rospy.loginfo("Goal Reached! Final yaw: %.2f rad", self.robot_yaw)
                    self.has_goal = False
                    self.position_reached = False
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