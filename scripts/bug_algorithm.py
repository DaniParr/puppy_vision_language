#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import Twist
from puppy_control.msg import Velocity
from sensor_msgs.msg import LaserScan

from puppy_mover import PuppyPiDirectDriver

class BugAlgorithm(PuppyPiDirectDriver):

    def __init__(self):

        super().__init__

        #SUBSCRIBER TO LIDAR
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.bug_state = "GO_TO_GOAL"  # Can be "GO_TO_GOAL" or "WALL_FOLLOW"

        self.OBSTACLE_THRESHOLD = 0.5  # meters to trigger obstacle avoidance
        self.WALL_FOLLOW_DIST = 0.4    # meters to maintain from the wall
        self.K_WALL = 2.0

        
        self.regions = {
            'front': float('inf'),
            'right': float('inf'),
            'left': float('inf')
        }

    def scan_callback(sel, msg):

        """Processes 360-degree LiDAR into 3 discrete regions."""
        ranges = msg.ranges
        num_readings = len(ranges)
        if num_readings == 0:
            return


        def get_min_dist(slice_arr):
            # Clean array of NaNs, Infs, and zeroes
            valid = [r for r in slice_arr if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max]
            return min(valid) if len(valid) > 0 else float('inf')

        # Dynamically calculate indices based on total readings (usually 360, 500, or 1000)
        # Front region: -20 to +20 degrees (combines the end and beginning of the array)
        front_idx = int((20.0 / 360.0) * num_readings)
        front_slice = ranges[-front_idx:] + ranges[:front_idx]
        
        # Right region: 250 to 290 degrees
        right_start = int((250.0 / 360.0) * num_readings)
        right_end = int((290.0 / 360.0) * num_readings)
        right_slice = ranges[right_start:right_end]
        
        # Left region: 70 to 110 degrees
        left_start = int((70.0 / 360.0) * num_readings)
        left_end = int((110.0 / 360.0) * num_readings)
        left_slice = ranges[left_start:left_end]

        # Update the regions state variable
        self.regions['front'] = get_min_dist(front_slice)
        self.regions['right'] = get_min_dist(right_slice)
        self.regions['left'] = get_min_dist(left_slice)

    
    def control_loop(self):

        #state_dict = {}

        while not rospy.is_shutdown():
            if not self.has_goal:
                self.RATE.sleep()
                continue
            
            # Calculate distance and angle to the target
            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            distance_error = math.sqrt(dx**2 + dy**2)
            angle_to_goal = math.atan2(dy, dx)
            heading_error = self.normalize_angle(angle_to_goal - self.robot_yaw)

            #Arrival check. Check if distance error is less than tolerance.

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
            
            # ==========================================
            # THE BUG 0 LOGIC 
            # ==========================================

            # STATE 1: GO TO GOAL
            if self.bug_state == "GO_TO_GOAL":
                
                # Trigger: Obstacle ahead! 
                if self.regions['front'] < self.OBSTACLE_THRESHOLD:
                    rospy.logwarn("Obstacle at %.2fm! Switching to WALL_FOLLOW.", self.regions['front'])
                    self.bug_state = "WALL_FOLLOW"
                    self.stop_robot()
                    self.RATE.sleep()
                    continue

                # Your original Phase 1 & 2 blending logic
                BLEND_START_DIST = 0.5
                blend = 1.0 - min(distance_error / BLEND_START_DIST, 1.0)
                final_yaw_error = self.normalize_angle(self.goal_yaw - self.robot_yaw)
                blended_heading_error = (1.0 - blend) * heading_error + blend * final_yaw_error

                # Pivot if facing away, drive if facing forward
                if abs(heading_error) > 0.2:
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

            # STATE 2: WALL FOLLOW
            elif self.bug_state == "WALL_FOLLOW":
                
                # Trigger: Line of sight restored (facing goal AND path is clear)
                if abs(heading_error) < 0.3 and self.regions['front'] > (self.OBSTACLE_THRESHOLD + 0.1):
                    rospy.loginfo("Line of sight clear! Resuming GO_TO_GOAL.")
                    self.bug_state = "GO_TO_GOAL"
                    self.stop_robot()
                    self.RATE.sleep()
                    continue

                # Keep a cautious forward speed while wall following
                velocity.x = self.MIN_LINEAR_SPEED + 2 
                
                # If we get stuck in an inside corner, turn left sharply in place
                if self.regions['front'] < self.OBSTACLE_THRESHOLD:
                    velocity.x = 0.0
                    velocity.yaw_rate = self.MAX_ANGULAR_SPEED
                else:
                    # Proportional control for tracking the right wall
                    error = self.regions['right'] - self.WALL_FOLLOW_DIST
                    
                    if math.isinf(self.regions['right']):
                        # We lost the wall (outside corner)! Turn right to wrap around it.
                        velocity.yaw_rate = -self.MAX_ANGULAR_SPEED
                    else:
                        # Steer to maintain exact distance
                        steer = error * self.K_WALL 
                        velocity.yaw_rate = self.apply_velocity_limits(
                            steer, self.MAX_ANGULAR_SPEED, self.MIN_ANGULAR_SPEED
                        )

            # Send the final calculated velocity to the hardware
            self.publish_velocity(velocity)
            self.RATE.sleep()
            

if __name__ == '__main__':
    try:
        driver = PuppyPiBugDriver()
        driver.control_loop()
    except rospy.ROSInterruptException:
        pass