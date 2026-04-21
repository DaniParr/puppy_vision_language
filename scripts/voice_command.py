import speech_recognition as sr
import sys
import re

def parse_and_print(text):
    pattern = r"(move|moved|go|drive|head)\s+(towards|toward|to)\s+(.*)"
    match = re.search(pattern, text)
    
    if match:
        action = "move"
        target = match.group(3).strip()
        
        if target:
            print(f"  -> MATCH FOUND! (Raw text: '{text}')")
            print(f"  -> INTENT EXTRACTED | Action: '{action}', Target: '{target}'")
        else:
            # Updated to show exactly what was heard when an object is missing
            print(f"  -> IGNORED: Heard '{text}', but no target object was specified.")
            
    else:
        # Updated to print the exact phrase that failed the regex pattern
        print(f"  -> NO MATCH: '{text}' did not contain a valid movement command.")


def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Keep the relaxed pause threshold so you don't get cut off
    recognizer.pause_threshold = 1.5 
    
    print("Calibrating microphone for ambient noise... please wait.")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=2.0)
    
    print("Calibration complete!")
    print("-" * 50)
    print("Ready! The robot will ONLY listen to 'move towards [object]'.")
    print("Try saying: 'Move towards the red couch' or 'Move towards the male person'.")
    print("Press Ctrl+C to stop the script.")
    print("-" * 50)

    try:
        while True:
            with microphone as source:
                print("\nListening...")
                try:
                    audio = recognizer.listen(source, timeout=5.0)
                    print("Processing...")
                    
                    text = recognizer.recognize_google(audio).lower()
                    print(f"Heard: '{text}'")
                    
                    # Run our strict parsing function
                    parse_and_print(text)
                    
                except sr.WaitTimeoutError:
                    pass  
                except sr.UnknownValueError:
                    print("  -> Could not understand the audio.")
                except sr.RequestError as e:
                    print(f"  -> API Request error: {e}")
                    
    except KeyboardInterrupt:
        print("\nExiting speech tester. Goodbye!")
        sys.exit(0)

if __name__ == '__main__':
    main()