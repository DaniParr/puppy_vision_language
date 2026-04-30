Robo Puppy Scan and Move final project: Steffano Cornejo, Daniel Parra

Robo Puppy sends http requests to Gemini in order to detect what the object we are looking for is,
and then the puppy uses FAST for depth perception in order to know the (X,Y) coordinate, with X being 
immediately in front, positive Y being to the right of the front facing part of the robot. This coordinate 
is the sent to the controller, which uses a 2D lidar to scan its enviornment, and then uses Hector SLAM to go 
from it's current coordinate to the coordinate that was passed in.

Contributions:

Steffano - Controller and Hector SLAM configuration based on Coordinate Passed in, making sure robot rotated and moved the correct amount 
based on the 2D lidar.
Daniel - Implemented the http request to gemini to allow the robo puppy to scan and identity the object. Also wrote the FAST algorithm to send
the depth and lateral coordinate to the controller.
