#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import speech_recognition as sr

class VoiceCommandNode:
    def __init__(self):
        rospy.init_node('voice_command_node', anonymous=True)
        
        # Publish directly to the topic your vision node is listening to
        self.prompt_pub = rospy.Publisher("/prompt", String, queue_size=10)
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        self.recognizer.pause_threshold = 0.8 
        
        rospy.loginfo("Calibrating microphone for ambient noise... please wait.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
            
        rospy.loginfo("Calibration complete! Ready to listen.")
        rospy.loginfo("-" * 50)

    def run(self):
        """Main listening loop tied to the ROS core."""
        while not rospy.is_shutdown():
            with self.microphone as source:
                try:
                    # Short timeout prevents the microphone from hanging indefinitely
                    audio = self.recognizer.listen(source, timeout=2.0, phrase_time_limit=10.0)
                    
                    # Send to Google STT
                    raw_text = self.recognizer.recognize_google(audio).lower()
                    rospy.loginfo(f"Heard: '{raw_text}'")
                    
                    # Publish immediately. Let the brain.py figure out what it means!
                    msg = String()
                    msg.data = raw_text
                    self.prompt_pub.publish(msg)
                    
                except sr.WaitTimeoutError:
                    # Normal behavior if the lab is quiet. Just silently loop.
                    pass  
                except sr.UnknownValueError:
                    # Heard a noise, but wasn't a word
                    pass 
                except sr.RequestError as e:
                    rospy.logerr(f"STT API Request error: {e}")

if __name__ == '__main__':
    try:
        node = VoiceCommandNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Exiting speech tester. Goodbye!")