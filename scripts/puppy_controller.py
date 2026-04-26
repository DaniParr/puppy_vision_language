#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PoseStamped, Point
from puppy_control.msg import Velocity, Pose

class PuppyController:
    def __init__(self):
        rospy.init_node('puppy_controller')

        # --- State ---
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.target_x = None
        self.target_y = None
        self.target_delta_yaw = 0.0  # z from /puppy_move

        # --- Tuning Parameters ---
        self.linear_speed      = rospy.get_param('~linear_speed',      0.15)  # m/s
        self.angular_speed     = rospy.get_param('~angular_speed',     0.5)   # rad/s
        self.goal_tolerance    = rospy.get_param('~goal_tolerance',    0.05)  # meters
        self.heading_tolerance = rospy.get_param('~heading_tolerance', 0.08)  # radians (~5°)
        self.backwards_thresh  = rospy.get_param('~backwards_thresh',  math.pi * 0.6)  # ~108°

        # --- ROS I/O ---
        self.pose_sub   = rospy.Subscriber('/slam_out_pose',      PoseStamped, self.pose_callback)
        self.target_sub = rospy.Subscriber('/puppy_move',         Point,       self.target_callback)
        self.vel_pub    = rospy.Publisher('/puppy_control/velocity', Velocity,  queue_size=10)
        self.pup_pose_pub = rospy.Publisher('/puppy_control/pose', Pose, queue_size=10)

        self.rate = rospy.Rate(20)  # 20 Hz
        rospy.loginfo("Puppy controller ready.")

        # Make puppy stand
        stand = Pose()
        stand.roll = math.radians(0)
        stand.pitch = math.radians(0)
        stand.yaw = 0.000
        stand.height = -10
        stand.x_shift = 0.4
        stand.stance_x = 0
        stand.stance_y = 0
        stand.run_time = 3

        self.pup_pose_pub.publish(stand)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def pose_callback(self, msg: PoseStamped):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_yaw = self.quat_to_yaw(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )

    def target_callback(self, msg: Point):
        self.target_x = msg.x
        self.target_y = msg.y
        self.target_delta_yaw = msg.z   # desired heading offset from x-axis (radians)
        rospy.loginfo(f"New target → x={msg.x:.2f}  y={msg.y:.2f}  Δyaw={msg.z:.2f} rad")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while not rospy.is_shutdown():
            if self.target_x is not None:
                self.control_step()
            self.rate.sleep()

    def control_step(self):
        # ── 1. Distance to target ─────────────────────────────────────
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = math.hypot(dx, dy)

        vel = Velocity()

        if distance < self.goal_tolerance:
            # ── Goal reached: apply final heading correction ───────────
            # target_delta_yaw is a delta from the global x-axis
            desired_final_yaw = self.normalize_angle(self.target_delta_yaw)
            yaw_error = self.normalize_angle(desired_final_yaw - self.current_yaw)

            if abs(yaw_error) > self.heading_tolerance:
                vel.angular_velocity = self.angular_speed * math.copysign(1, yaw_error)
            else:
                vel.angular_velocity = 0.0
                rospy.loginfo_throttle(2.0, "Goal reached and heading achieved.")

            vel.linear_velocity = 0.0
            self.vel_pub.publish(vel)
            return

        # ── 2. Angle from current position toward target ──────────────
        angle_to_target = math.atan2(dy, dx)   # global frame
        heading_error   = self.normalize_angle(angle_to_target - self.current_yaw)

        # ── 3. Decide: walk forward or backward? ──────────────────────
        #   If |heading_error| > backwards_thresh it's cheaper to reverse.
        if abs(heading_error) > self.backwards_thresh:
            # Flip: face the opposite way and drive in reverse
            heading_error  = self.normalize_angle(heading_error - math.pi)
            linear_sign    = -1.0
        else:
            linear_sign    = 1.0

        # ── 4. Rotate first, then walk ────────────────────────────────
        if abs(heading_error) > self.heading_tolerance:
            # Still turning — publish rotation only
            vel.angular_velocity = self.angular_speed * math.copysign(1, heading_error)
            vel.linear_velocity  = 0.0
        else:
            # Heading good — walk toward target
            vel.angular_velocity = 0.0
            vel.linear_velocity  = linear_sign * self.linear_speed

        self.vel_pub.publish(vel)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def quat_to_yaw(qx, qy, qz, qw) -> float:
        """Extract yaw (rotation about Z) from a quaternion."""
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Wrap angle to [-π, π]."""
        while angle >  math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle


if __name__ == '__main__':
    try:
        PuppyController().run()
    except rospy.ROSInterruptException:
        pass
