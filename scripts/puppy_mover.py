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
        self.STOP_DISTANCE = 0.05
        self.RATE = rospy.Rate(10)
        self.pose_received = False

        # Velocity Limits (raw hardware units)
        self.MAX_LINEAR_SPEED = 15
        self.MAX_ANGULAR_SPEED = 0.5
        
        self.MIN_LINEAR_SPEED = 7
        self.MIN_ANGULAR_SPEED = 0.2

        # Proportional Control Gains
        # K_LINEAR scaled to map meter distances to raw hardware speed units
        self.K_LINEAR = 50.0
        self.K_ANGULAR = 1.0
        
        # Tolerances
        self.DIST_TOLERANCE = 0.01   # meters — reachable given hardware speed
        self.YAW_TOLERANCE = 0.05    # radians

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

        # --- ROS INTERFACES ---
        self.pose_sub = rospy.Subscriber('/slam_out_pose', PoseStamped, self.pose_callback)
        self.target_sub = rospy.Subscriber('/puppy_move', Point, self.target_callback)
        self.pup_velocity_pub = rospy.Publisher('/puppy_control/velocity', Velocity, queue_size=10)
        self.pup_pose_pub = rospy.Publisher('/puppy_control/pose', Pose, queue_size=10)
        self.pup_pose_pub.publish(STAND)

        rospy.on_shutdown(self.stop_robot)
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
            self.has_goal = True
            return

        # --- CASE 3: Drive toward target (x!=0, y optional) ---
        bearing_to_target = math.atan2(depth_y, depth_x)
        total_distance = math.sqrt(depth_x**2 + depth_y**2)
        travel_distance = total_distance - self.STOP_DISTANCE

        approach_yaw = self.normalize_angle(self.robot_yaw + bearing_to_target)

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

            BLEND_START_DIST = 0.5
            blend = 1.0 - min(distance_error / BLEND_START_DIST, 1.0)
            final_yaw_error = self.normalize_angle(self.goal_yaw - self.robot_yaw)
            blended_heading_error = (1.0 - blend) * heading_error + blend * final_yaw_error

            # Debug: log distance and heading so you can monitor convergence
            rospy.loginfo_throttle(
                0.5,  # log at most every 0.5 seconds to avoid spam
                "dist_err=%.3fm heading_err=%.2frad robot=(%.2f,%.2f) goal=(%.2f,%.2f)",
                distance_error, heading_error,
                self.robot_x, self.robot_y,
                self.goal_x, self.goal_y
            )

            velocity = Velocity()

            # Phase 1: Not facing the target direction — pivot first
            if distance_error > self.DIST_TOLERANCE and abs(heading_error) > 0.2:
                velocity.yaw_rate = self.K_ANGULAR * blended_heading_error
                velocity.yaw_rate = self.apply_velocity_limits(
                    velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                )

            # Phase 2: Facing the target — walk forward while correcting heading
            elif distance_error > self.DIST_TOLERANCE:
                linear_speed = self.K_LINEAR * distance_error
                linear_speed = self.apply_velocity_limits(
                    linear_speed, self.MAX_LINEAR_SPEED, self.MIN_LINEAR_SPEED
                )
                velocity.x = linear_speed

                velocity.yaw_rate = self.K_ANGULAR * blended_heading_error
                velocity.yaw_rate = self.apply_velocity_limits(
                    velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                )

            # Phase 3: At the destination — rotate to final goal_yaw
            else:
                if abs(final_yaw_error) > self.YAW_TOLERANCE:
                    velocity.yaw_rate = self.K_ANGULAR * final_yaw_error
                    velocity.yaw_rate = self.apply_velocity_limits(
                        velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                    )
                else:
                    rospy.loginfo("Goal Reached! Final yaw: %.2f rad", self.robot_yaw)
                    self.has_goal = False
                    self.stop_robot()
                    continue  # stop_robot already published, skip publish_velocity below

            self.publish_velocity(velocity)
            self.RATE.sleep()


if __name__ == '__main__':
    try:
        driver = PuppyPiDirectDriver()
        driver.control_loop()
    except rospy.ROSInterruptException:
        pass