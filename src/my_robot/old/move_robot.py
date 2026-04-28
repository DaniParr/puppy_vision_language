#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import time
import math

rospy.init_node('move_robot')
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
time.sleep(1)

def move(distance, speed=0.5):
    msg = Twist()
    duration = abs(distance) / speed
    msg.linear.x = speed if distance > 0 else -speed
    print(f"Moving {'forward' if distance > 0 else 'backward'} {abs(distance)}m...")
    start = time.time()
    while time.time() - start < duration:
        pub.publish(msg)
        time.sleep(0.1)
    pub.publish(Twist())
    print("Done!")

def turn(degrees, speed=0.5):
    msg = Twist()
    radians = abs(degrees) * (math.pi / 180)
    duration = radians / speed
    msg.angular.z = speed if degrees > 0 else -speed
    print(f"Turning {'left' if degrees > 0 else 'right'} {abs(degrees)} degrees...")
    start = time.time()
    while time.time() - start < duration:
        pub.publish(msg)
        time.sleep(0.1)
    pub.publish(Twist())
    print("Done!")

def print_help():
    print("""
Commands:
  move <distance>        move forward (positive) or backward (negative) in meters
  turn <degrees>         turn left (positive) or right (negative) in degrees
  help                   show this help message
  quit                   exit

Examples:
  move 2.0
  move -1.0
  turn 90
  turn -45
    """)

print("Robot controller ready!")
print_help()

while not rospy.is_shutdown():
    try:
        user_input = input(">> ").strip().split()

        if not user_input:
            continue

        command = user_input[0].lower()

        if command == "quit":
            print("Bye!")
            break

        elif command == "help":
            print_help()

        elif command == "move":
            if len(user_input) < 2:
                print("Usage: move <distance>")
            else:
                move(float(user_input[1]))

        elif command == "turn":
            if len(user_input) < 2:
                print("Usage: turn <degrees>")
            else:
                turn(float(user_input[1]))

        else:
            print(f"Unknown command: '{command}'. Type 'help' for available commands.")

    except KeyboardInterrupt:
        print("\nBye!")
        break
    except ValueError:
        print("Invalid value. Please enter a number.")
