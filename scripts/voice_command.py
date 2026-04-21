
import speech_recognition as sr
     
def main():
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        print("Calibrating microphone for ambient noise... please wait.")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=2.0)
        
        print("Calibration complete!")
        print("-" * 50)
        print("Ready! Say something like 'Move to the red couch'.")
        print("Press Ctrl+C to stop the script.")
        print("-" * 50)

        try:
            while True:
                with microphone as source:
                    print("\nListening...")
                    try:
                        # Capture audio
                        audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=3.0)
                        print("Processing...")
                        
                        # Transcribe
                        text = recognizer.recognize_google(audio).lower()
                        print(f"Heard: '{text}'")
                        
                        # Parse
                        #parse_and_print(text)
                        
                    except sr.WaitTimeoutError:
                        pass  # Normal timeout, just loops back to "Listening..."
                    except sr.UnknownValueError:
                        print("  -> Could not understand the audio.")
                    except sr.RequestError as e:
                        print(f"  -> API Request error: {e}")
                        
        except KeyboardInterrupt:
            print("\nExiting speech tester. Goodbye!")
            sys.exit(0)

if __name__ == '__main__':
    main()   
