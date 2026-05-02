Robo Puppy Scan and Move final project: Steffano Cornejo, Daniel Parra, Jorge Ramirez, Hector Borjas

Robo Puppy sends http requests to Gemini in order to detect what the object we are looking for is,
and then the puppy uses FAST for depth perception in order to know the (X,Y) coordinate, with X being 
immediately in front, positive Y being to the right of the front facing part of the robot. This coordinate 
is the sent to the controller, which uses a 2D lidar to scan its environment, and then uses Hector SLAM to go 
from it's current coordinate to the coordinate that was passed in. There is an implementation of the Bug0 algorithm for obstacle avoidance, which is only functional in our gazebo simulation.

Contributions:

Steffano - Controller and Hector SLAM configuration based on Coordinate Passed in, making sure robot rotated and moved the correct amount 
based on the 2D lidar.
Daniel - Implemented the http request to gemini to allow the robo puppy to scan and identity the object. Also wrote the FAST algorithm to send the depth and lateral coordinate to the controller.
Jorge - Developed bug0 algorithm and set up a ROS package with the nodes required for testing bug0 functionality within a Gazebo simulation (prompt_to_point.py and puppy_gazebo_bridge.py). Simulation Package is available within the branch 'gazebo_test'
Hector - Developed Voice Detection algorithm and also contributed to the bug_algorithm and the puppy_gazebo_bridge.py. Also worked on trying to get the obstacle avoidance to function on the PuppyPi

Link to Repository: https://github.com/DaniParr/puppy_vision_language/tree/working-mvp 

Video of Simulation: https://youtu.be/vMnWlSkYFik
