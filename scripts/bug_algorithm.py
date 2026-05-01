#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import Twist
from puppy_control.msg import Velocity
from sensor_msgs.msg import LaserScan
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from puppy_mover import PuppyPiDirectDriver

class BugAlgorithm(PuppyPiDirectDriver):

    def __init__(self):
        super().__init__()

        self.K_ANGULAR = 30.0
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.bug_state = "GO_TO_GOAL"

        self.clear_count = 0
        self.CLEAR_COUNT_THRESHOLD = 10

        self.OBSTACLE_THRESHOLD = 0.25
        self.DESIRED_WALL_DIST  = 0.20
        self.avoid_target_yaw   = 0.0
        self.K_WALL = 2.0

        self.regions = {
            'front': float('inf'),
            'right': float('inf'),
            'left':  float('inf'),
        }

    def scan_callback(self, msg):
        # ... (keep yours exactly as-is, it works)
        ranges = msg.ranges
        num_readings = len(ranges)
        if num_readings == 0:
            return

        def get_min_dist(slice_arr):
            valid = [r for r in slice_arr if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max]
            return min(valid) if len(valid) > 0 else float('inf')

        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        front_center = int(round((0.0 - angle_min) / angle_increment))
        right_center = int(round((-math.pi/2 - angle_min) / angle_increment))
        left_center = int(round((math.pi/2 - angle_min) / angle_increment))

        spread = int(round((20.0 * math.pi / 180) / angle_increment))

        def safe_slice(center, spread):
            start = max(0, center - spread)
            end = min(num_readings, center + spread)
            return list(ranges[start:end])

        self.regions['front'] = get_min_dist(safe_slice(front_center, spread))
        self.regions['right'] = get_min_dist(safe_slice(right_center, spread))
        self.regions['left'] = get_min_dist(safe_slice(left_center, spread))

        rospy.loginfo(f"front={self.regions['front']:.2f} right={self.regions['right']:.2f} left={self.regions['left']:.2f}")

    def _compute_drive_velocity(self, distance_error, heading_error, final_yaw_error):
        """
        Mirrors the parent's 3-phase logic EXACTLY — the only proven-working
        drive code for this robot. Returns a Velocity message.
        """
        COAST_DISTANCE = 0.08

        velocity = Velocity()

        if not self.position_reached:

            # Phase 1: Severely off-course — pure pivot with x coupling
            if (distance_error > COAST_DISTANCE
                    and abs(heading_error) > math.pi / 2
                    and self.goal_x != self.robot_x):

                velocity.yaw_rate = self.apply_velocity_limits(
                    self.K_ANGULAR * heading_error,
                    self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                )
                velocity.x = abs(velocity.yaw_rate) * 25  # quadruped coupling

            # Phase 2: Walk toward goal
            else:
                velocity.x = self.apply_velocity_limits(
                    self.K_LINEAR * distance_error,
                    self.MAX_LINEAR_SPEED, self.MIN_LINEAR_SPEED
                )
                # Coast: near the goal, lock to final yaw instead of tracking XY angle
                steer_error = final_yaw_error if distance_error < COAST_DISTANCE else heading_error
                velocity.yaw_rate = self.apply_velocity_limits(
                    self.K_ANGULAR * steer_error,
                    self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                )

        # Phase 3: Position reached — rotate to final yaw
        else:
            if abs(final_yaw_error) > self.YAW_TOLERANCE:
                velocity.yaw_rate = self.apply_velocity_limits(
                    self.K_ANGULAR * final_yaw_error,
                    self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                )
                velocity.x = abs(velocity.yaw_rate) * 25  # quadruped coupling
            else:
                return None  # signals "goal fully reached"

        return velocity

    def control_loop(self):
        while not rospy.is_shutdown():
            if not self.has_goal:
                self.RATE.sleep()
                continue

            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            distance_error    = math.sqrt(dx**2 + dy**2)
            angle_to_goal     = math.atan2(dy, dx)
            heading_error     = self.normalize_angle(angle_to_goal - self.robot_yaw)
            final_yaw_error   = self.normalize_angle(self.goal_yaw - self.robot_yaw)

            # Hard latch — same as parent
            if distance_error <= self.DIST_TOLERANCE:
                self.position_reached = True

            # ─── STATE: GO_TO_GOAL ───────────────────────────────────────────
            if self.bug_state == "GO_TO_GOAL":

                if self.regions['front'] < self.OBSTACLE_THRESHOLD and not self.position_reached:
                    rospy.logwarn("Obstacle! Starting 90° left turn.")
                    self.avoid_target_yaw = self.normalize_angle(self.robot_yaw + math.pi / 2)
                    self.bug_state = "AVOID_TURN"
                    self.stop_robot()
                    self.RATE.sleep()
                    continue

                velocity = self._compute_drive_velocity(
                    distance_error, heading_error, final_yaw_error
                )

                if velocity is None:
                    rospy.loginfo("Goal reached!")
                    self.has_goal = False
                    self.position_reached = False
                    self.stop_robot()
                    self.RATE.sleep()
                    continue

            # ─── STATE: AVOID_TURN ───────────────────────────────────────────
            elif self.bug_state == "AVOID_TURN":
                turn_error = self.normalize_angle(self.avoid_target_yaw - self.robot_yaw)

                if abs(turn_error) < 0.15:
                    rospy.loginfo("Turn done. Wall-following.")
                    self.bug_state = "WALL_FOLLOW"
                    self.stop_robot()
                    self.RATE.sleep()
                    continue

                velocity = Velocity()
                velocity.yaw_rate = self.MAX_ANGULAR_SPEED          # turn left
                velocity.x        = velocity.yaw_rate * 25          # quadruped coupling!

            # ─── STATE: WALL_FOLLOW ──────────────────────────────────────────
            elif self.bug_state == "WALL_FOLLOW":
                # Obstacle is now on the RIGHT — follow it
                wall_dist  = self.regions['right']
                front_clear = self.regions['front'] > self.OBSTACLE_THRESHOLD + 0.1
                right_open  = wall_dist > self.OBSTACLE_THRESHOLD + 0.15

                if front_clear and right_open:
                    self.clear_count += 1
                    if self.clear_count >= self.CLEAR_COUNT_THRESHOLD:
                        rospy.loginfo("Corner cleared. Back to GO_TO_GOAL.")
                        self.bug_state  = "GO_TO_GOAL"
                        self.clear_count = 0
                        self.stop_robot()
                        self.RATE.sleep()
                        continue
                else:
                    self.clear_count = 0

                # Proportional wall-following — keep right wall at DESIRED_WALL_DIST
                wall_error = self.DESIRED_WALL_DIST - wall_dist  # + = too close, − = too far
                velocity = Velocity()
                velocity.x = self.MIN_LINEAR_SPEED * 2
                velocity.yaw_rate = self.apply_velocity_limits(
                    self.K_WALL * wall_error,
                    self.MAX_ANGULAR_SPEED, 0.0  # zero minimum — don't force a turn
                )

            self.publish_velocity(velocity)
            self.RATE.sleep()

if __name__ == '__main__':
    try:
        driver = BugAlgorithm()
        driver.control_loop()
    except rospy.ROSInterruptException:
        pass