from pynput import keyboard
import time 
import numpy as np 

def gripper_control():
        print("Now actively controlling gripper. Press a and d to control gripper and q to move on")
        gripper_active = True
        change_gripper_by = 0.05 #rad
        period = 0.05 #s
        current_val = 0 
        
        # Make sure keys_pressed is initialized
        keys_pressed = {}

        def on_press(key):
            try:
                keys_pressed[key.char] = True
            except AttributeError:
                # Special keys
                if key == keyboard.Key.up:
                    keys_pressed['up'] = True
                elif key == keyboard.Key.down:
                    keys_pressed['down'] = True
                elif key == keyboard.Key.left:
                    keys_pressed['left'] = True
                elif key == keyboard.Key.right:
                    keys_pressed['right'] = True
        
        def on_release(key):
            try:
                keys_pressed[key.char] = False
            except AttributeError:
                # Special keys
                if key == keyboard.Key.up:
                    keys_pressed['up'] = False
                elif key == keyboard.Key.down:
                    keys_pressed['down'] = False
                elif key == keyboard.Key.left:
                    keys_pressed['left'] = False
                elif key == keyboard.Key.right:
                    keys_pressed['right'] = False
        
        # Start keyboard listener
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        
        # Small delay to ensure listener is active
        time.sleep(0.1)

        while gripper_active:
            # Reset gripper_change for each iteration
            gripper_change = 0
            
            # Control gripper with a/d keys
            if keys_pressed.get('d', False):
                gripper_change += change_gripper_by #rad
            if keys_pressed.get('a', False):
                gripper_change -= change_gripper_by #rad

            #make sure that the gripper cannot be commanded to overactuate
            if current_val + gripper_change > np.pi/2 or current_val + gripper_change < -np.pi/2:
                gripper_change = 0
            
            current_val += gripper_change

            print(f"Current value = {np.round(current_val, 4)}", end="\r")

            #Check for the quit message
            if keys_pressed.get('q', False):
                gripper_active = False

            time.sleep(period)

        # Clean up the listener when done
        listener.stop()
        print("Gripper control has been quit. Moving on...")
        return 
    

print("Before Calling Gripper_Control")
gripper_control()
print("After Calling Gripper_Control")q