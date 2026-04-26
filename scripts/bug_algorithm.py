from sensor_msgs.msg import LaserScan

from puppy_mover import PuppyPiDirectDriver
class BugAlgorithm(PuppyPiDirectDriver):

    def __init__(self):

        super().__init__

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
        


    def scan_callback(self):

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

        state_dict = {}

        while not rospy.is_shutdown():
            if not self.has_goal:
                self.RATE.sleep()
                continue
                
            # (Paste the Bug0 State Machine logic here)
            pass


if __name__ == '__main__':
    try:
        driver = PuppyPiBugDriver()
        driver.control_loop()
    except rospy.ROSInterruptException:
        pass