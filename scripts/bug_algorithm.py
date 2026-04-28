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
        self.CLEAR_COUNT_THRESHOLD = 10  # must be clear for 10 readings before resuming

        self.OBSTACLE_THRESHOLD = 0.5
        self.WALL_FOLLOW_DIST = 0.4
        self.K_WALL = 2.0

        self.regions = {
            'front': float('inf'),
            'right': float('inf'),
            'left': float('inf')
        }

    def scan_callback(self, msg):
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

    def control_loop(self):
        while not rospy.is_shutdown():
            if not self.has_goal:
                self.RATE.sleep()
                continue

            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            distance_error = math.sqrt(dx**2 + dy**2)
            angle_to_goal = math.atan2(dy, dx)
            heading_error = self.normalize_angle(angle_to_goal - self.robot_yaw)

            if distance_error <= self.DIST_TOLERANCE:
                final_yaw_error = self.normalize_angle(self.goal_yaw - self.robot_yaw)
                if abs(final_yaw_error) > self.YAW_TOLERANCE:
                    velocity = Velocity()
                    velocity.yaw_rate = self.K_ANGULAR * final_yaw_error
                    velocity.yaw_rate = self.apply_velocity_limits(
                        velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                    )
                    self.publish_velocity(velocity)
                else:
                    rospy.loginfo("Goal Reached! Awaiting next command.")
                    self.has_goal = False
                    self.stop_robot()
                self.RATE.sleep()
                continue

            velocity = Velocity()

            if self.bug_state == "GO_TO_GOAL":
                if self.regions['front'] < self.OBSTACLE_THRESHOLD:
                    rospy.logwarn("Obstacle at %.2fm! Switching to WALL_FOLLOW.", self.regions['front'])
                    self.bug_state = "WALL_FOLLOW"
                    self.stop_robot()
                    self.RATE.sleep()
                    continue

                BLEND_START_DIST = 0.5
                blend = 1.0 - min(distance_error / BLEND_START_DIST, 1.0)
                final_yaw_error = self.normalize_angle(self.goal_yaw - self.robot_yaw)
                blended_heading_error = (1.0 - blend) * heading_error + blend * final_yaw_error

                if abs(heading_error) > 0.5:
                    velocity.yaw_rate = self.K_ANGULAR * blended_heading_error
                    velocity.yaw_rate = self.apply_velocity_limits(
                        velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                    )
                else:
                    linear_speed = self.K_LINEAR * distance_error
                    velocity.x = self.apply_velocity_limits(
                        linear_speed, self.MAX_LINEAR_SPEED, self.MIN_LINEAR_SPEED
                    )
                    velocity.yaw_rate = self.K_ANGULAR * blended_heading_error
                    velocity.yaw_rate = self.apply_velocity_limits(
                        velocity.yaw_rate, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                    )

            elif self.bug_state == "WALL_FOLLOW":
                if self.regions['front'] > (self.OBSTACLE_THRESHOLD + 0.1) and self.regions['left'] > (self.OBSTACLE_THRESHOLD + 0.1):
                    self.clear_count+=1
                    if self.clear_count >= self.CLEAR_COUNT_THRESHOLD:
                        rospy.loginfo("Path clear! Resuming GO_TO_GOAL.")
                        self.bug_state = "GO_TO_GOAL"
                        self.clear_count = 0
                        self.stop_robot()
                        self.RATE.sleep()
                        continue
                else:
                    self.clear_count = 0

                if self.regions['front'] < self.OBSTACLE_THRESHOLD:
                    rospy.loginfo("Turning left to clear obstacle...")
                    velocity.x = 0.0
                    velocity.yaw_rate = 10.0
                else:
                    rospy.loginfo("Following wall...")
                    velocity.x = self.MIN_LINEAR_SPEED*3
                    if math.isinf(self.regions['right']):
                        velocity.yaw_rate = -self.MAX_ANGULAR_SPEED
                    else:
                        error = self.regions['right'] - self.WALL_FOLLOW_DIST
                        steer = error * self.K_WALL
                        velocity.yaw_rate = self.apply_velocity_limits(
                            steer, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                        )

            self.publish_velocity(velocity)
            self.RATE.sleep()


if __name__ == '__main__':
    try:
        driver = BugAlgorithm()
        driver.control_loop()
    except rospy.ROSInterruptException:
        pass