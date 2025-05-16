from flask import Flask, request, jsonify
import lgpio
import datetime
import threading
import time
import sys # Import sys for error output

# --- Raspberry Pi GPIO Configuration (using lgpio BCM numbering) ---
# Verify the correct BCM pin numbers for your setup (matches BOARD in comments)
# Ensure these pins are connected correctly to your motor driver (e.g., L298N)
# !! NOTE: Based on your tests, the actual effect of these pins might differ from their intended label !!
Motor_R1_Pin = 23  # BCM 23 - Intended: Right Motor Forward. Actual: Right Motor Forward (根據測試)
Motor_R2_Pin = 24  # BCM 24 - Intended: Right Motor Backward. Actual: Does not move (根據測試)
Motor_L1_Pin = 17  # BCM 17 - Intended: Left Motor Forward. Actual: Left Motor Backward (根據測試)
Motor_L2_Pin = 27  # BCM 27 - Intended: Left Motor Backward. Actual: Does not move (根據測試)

# GPIO handle variable, initialized to None
h = None
# Flag to check if GPIO was successfully initialized
gpio_initialized = False

# --- Initialize GPIO Pins ---
try:
    print("Attempting to initialize GPIO...")
    # Open the GPIO chip (usually chip 0 on Raspberry Pi)
    h = lgpio.gpiochip_open(0)
    if h < 0: # lgpio.gpiochip_open returns negative on error
        print(f"Error: Failed to open GPIO chip, code {h} ({lgpio.error_text(h)})", file=sys.stderr)
        # Exit the script if GPIO chip cannot be opened
        sys.exit(1)

    print("GPIO chip opened successfully. Attempting to claim pins as outputs...")
    pins_to_claim = [Motor_R1_Pin, Motor_R2_Pin, Motor_L1_Pin, Motor_L2_Pin]
    claimed_pins = [] # Keep track of successfully claimed pins

    for pin in pins_to_claim:
        # Claim pin as output and set initial state to 0 (low)
        ret = lgpio.gpio_claim_output(h, pin, 0)
        if ret < 0:
            print(f"Error: Failed to claim GPIO pin BCM {pin}, reason: '{lgpio.error_text(ret)}' (e.g., GPIO busy)", file=sys.stderr)
            # If claiming fails, mark initialization as failed and break the loop
            gpio_initialized = False
            break # Exit the for loop
        else:
            print(f"Successfully claimed GPIO pin BCM {pin} as output.")
            claimed_pins.append(pin) # Add successfully claimed pin to list

    # After the loop, check if all pins were claimed (no break occurred)
    if len(claimed_pins) == len(pins_to_claim):
        gpio_initialized = True
        print("All required GPIO pins claimed successfully.")
    else:
         # If some pins failed to claim (break occurred), free any pins that *were* claimed
        print("Some GPIO pins failed to claim. Freeing already claimed pins...")
        for pin in claimed_pins:
             lgpio.gpio_free(h, pin)
        # Close the GPIO chip if it was opened
        if h is not None and h >= 0:
             lgpio.gpiochip_close(h)
             h = None # Set handle to None to prevent closing again in finally
        print("GPIO initialization failed. Please check if pins are already in use by another process.")
        # Exit the script because GPIO is not ready
        sys.exit(1)


except Exception as e:
    # Catch any other unexpected errors during initialization
    print(f"An unexpected error occurred during GPIO initialization: {e}", file=sys.stderr)
    gpio_initialized = False
    # Ensure GPIO chip is closed if it was opened
    if h is not None and h >= 0:
        lgpio.gpiochip_close(h)
        h = None
    sys.exit(1)

# --- Motor Control Functions ---

# Function to stop all motors
def stop():
    if not gpio_initialized or h < 0:
        # print("GPIO not initialized, cannot execute stop command.", file=sys.stderr) # avoid excessive printing on repeated calls
        return
    print("Stopping all motors...")
    try:
        # Set all motor control pins to low
        for pin in [Motor_R1_Pin, Motor_R2_Pin, Motor_L1_Pin, Motor_L2_Pin]:
            lgpio.gpio_write(h, pin, 0)
    except Exception as e:
        print(f"Error writing to GPIO during stop: {e}", file=sys.stderr)


# Helper function to run a single motor action for a duration and then stop
# Used by up, down, left, right actions with specific or default duration
def run_single_motor_action_for_duration(action_func, duration):
    """
    Runs a single motor action function for a specified duration, then calls stop.
    Intended to be run in a separate thread.
    """
    if not gpio_initialized or h < 0:
        print("GPIO not initialized, cannot run motor action.", file=sys.stderr)
        return

    try:
        # Execute the provided motor action function (e.g., forward, backward)
        action_func()
        # Wait for the specified duration
        time.sleep(duration)
    except Exception as e:
         print(f"Error during single motor action ({action_func.__name__}): {e}", file=sys.stderr)
    finally:
        # Ensure motors are stopped after the action/duration
        stop()

# Function to move both wheels forward
def forward():
    print("Motors: Forward (Intended: Both wheels forward)")
    lgpio.gpio_write(h, Motor_R1_Pin, 1) # 右輪前進控制腳位
    lgpio.gpio_write(h, Motor_R2_Pin, 0)
    lgpio.gpio_write(h, Motor_L1_Pin, 1) # 左輪前進控制腳位
    lgpio.gpio_write(h, Motor_L2_Pin, 0)

# Function to move both wheels backward
def backward():
    print("Motors: Backward (Intended: Both wheels backward)")
    lgpio.gpio_write(h, Motor_R1_Pin, 0)
    lgpio.gpio_write(h, Motor_R2_Pin, 1) # 右輪後退控制腳位
    lgpio.gpio_write(h, Motor_L1_Pin, 0)
    lgpio.gpio_write(h, Motor_L2_Pin, 1) # 左輪後退控制腳位

# Function to spin left (Right wheel forward, Left wheel backward)
def turn_left():
    print("Motors: Turn Left (Intended: Spin left)")
    # Intended standard spin left: Right wheel forward, Left wheel backward
    lgpio.gpio_write(h, Motor_R1_Pin, 1) # 右輪前進
    lgpio.gpio_write(h, Motor_R2_Pin, 0)
    lgpio.gpio_write(h, Motor_L1_Pin, 0)
    lgpio.gpio_write(h, Motor_L2_Pin, 1) # 左輪後退

# Function to spin right (Left wheel forward, Right wheel backward)
def turn_right():
    print("Motors: Turn Right (Intended: Spin right)")
    # Intended standard spin right: Left wheel forward, Right wheel backward
    lgpio.gpio_write(h, Motor_R1_Pin, 0)
    lgpio.gpio_write(h, Motor_R2_Pin, 1) # 右輪後退
    lgpio.gpio_write(h, Motor_L1_Pin, 1) # 左輪前進
    lgpio.gpio_write(h, Motor_L2_Pin, 0)

# Custom function for 'thumb up' gesture: spin in place for 5 seconds
# Moved this function definition BEFORE receive_command
def spin_in_place_for_duration(duration):
    """
    Spins the robot in place (one side forward, one side backward) for a duration.
    """
    if not gpio_initialized or h < 0:
        print("GPIO not initialized, cannot spin.", file=sys.stderr)
        return

    print(f"Executing spin in place for {duration} seconds...")
    try:
        # Intended: Right motor forward, Left motor backward for counter-rotation
        # Based on *intended* pin labels
        lgpio.gpio_write(h, Motor_R1_Pin, 1) # Right Forward ON
        lgpio.gpio_write(h, Motor_R2_Pin, 0) # Right Backward OFF
        lgpio.gpio_write(h, Motor_L1_Pin, 0) # Left Forward OFF
        lgpio.gpio_write(h, Motor_L2_Pin, 1) # Left Backward ON

        time.sleep(duration)
    except Exception as e:
        print(f"Error during spin_in_place_for_duration: {e}", file=sys.stderr)
    finally:
        stop()

# Custom function for 'thumb down' gesture: turn ~180 degrees then move forward
# Moved this function definition BEFORE receive_command
def turn_180_then_move(turn_duration, move_duration):
    """
    Executes a turn (using spin logic) for a duration, stops briefly, then moves forward.
    Approximates a 180-degree turn followed by forward movement.
    """
    if not gpio_initialized or h < 0:
        print("GPIO not initialized, cannot execute turn and move sequence.", file=sys.stderr)
        return

    print(f"Executing turn (~180 deg for {turn_duration}s) then move forward for {move_duration}s...")
    try:
        # Step 1: Turn (using intended spin logic: R=F, L=B for one direction)
        lgpio.gpio_write(h, Motor_R1_Pin, 1) # Right Forward ON
        lgpio.gpio_write(h, Motor_R2_Pin, 0) # Right Backward OFF
        lgpio.gpio_write(h, Motor_L1_Pin, 0) # Left Forward OFF
        lgpio.gpio_write(h, Motor_L2_Pin, 1) # Left Backward ON # Spin in place direction (intended)
        time.sleep(turn_duration)

        # Optional brief stop between actions
        stop()
        time.sleep(0.1) # Small pause

        # Step 2: Move forward (using intended forward logic)
        forward() # Use the existing forward function (should move both wheels forward, intended)
        time.sleep(move_duration)

    except Exception as e:
        print(f"Error during turn_180_then_move: {e}", file=sys.stderr)
    finally:
        stop()


# --- Other Event Handlers (unchanged as requested) ---

# These handlers are for events like alerts, logging, etc., not motor control.
def handle_alert(objects, timestamp):
    print(f"[{timestamp}] ALERT triggered with: {objects}")

def handle_record(objects, timestamp):
    print(f"[{timestamp}] RECORD triggered with: {objects}")

def handle_log(objects, timestamp):
    print(f"[{timestamp}] LOG triggered with: {objects}")
    log_entry = f"[{timestamp}] Event: Detection, Objects: {', '.join(objects)}, Action: log\n"
    try:
        with open("detection_log.txt", "a") as f:
            f.write(log_entry)
    except IOError as e:
        print(f"Error writing to log file: {e}", file=sys.stderr)

def handle_notify(objects, timestamp):
    print(f"[{timestamp}] NOTIFY triggered with: {objects}")

def handle_light_on(objects, timestamp):
    print(f"[{timestamp}] LIGHT_ON triggered with: {objects}")
    # Placeholder for code to turn on a light
    pass # Implement your logic here

def handle_buzzer_on(objects, timestamp):
    print(f"[{timestamp}] BUZZER_ON triggered with: {objects}")
    # Placeholder for code to turn on a buzzer
    pass # Implement your logic here

def handle_none(objects, timestamp):
    print(f"[{timestamp}] NONE action received with: {objects}")

def handle_unknown(action, objects, timestamp):
    print(f"[{timestamp}] UNKNOWN action '{action}' with: {objects}")

# --- Flask App Setup ---

app = Flask(__name__)

HOST_IP = '0.0.0.0'
HOST_PORT = 5000

@app.route('/command', methods=['POST'])
def receive_command():
    if not request.json:
        return jsonify({"status": "error", "message": "Request must be JSON"}), 415

    data = request.json
    # Ensure action is lowercase for consistent comparison
    action = data.get("action", "").lower()
    objects = data.get("objects", [])
    # Get duration, defaulting to 3 seconds if not provided
    motor_duration = data.get("duration", 3) # Default duration changed to 3 seconds
    timestamp = data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print(f"[{timestamp}] Received command: action='{action}', objects={objects}, duration={motor_duration}")

    # Check if GPIO is initialized before attempting any motor control
    if not gpio_initialized or h < 0:
         print(f"[{timestamp}] Received command {action}, but GPIO is not initialized. Cannot execute motor commands.", file=sys.stderr)
         # Return an error response if GPIO is not ready for motor commands
         if action in ['up', 'down', 'left', 'right', 'stop', 'thumb up', 'thumb down']:
            return jsonify({"status": "error", "message": "GPIO not initialized, cannot execute motor command."}), 500


    # --- Handle Motor Control Actions ---
    if action in ['up', 'down', 'left', 'right', 'stop', 'thumb up', 'thumb down']:
        if action == 'up':
            # Forward for 2 seconds (Intended, but currently causes spin due to wiring)
            thread = threading.Thread(target=run_single_motor_action_for_duration, args=(forward, 2.0)) # Explicitly 2.0s
            thread.start()
        elif action == 'down':
            # Backward for 2 seconds (Intended, but likely no movement due to wiring)
            thread = threading.Thread(target=run_single_motor_action_for_duration, args=(backward, 2.0)) # Explicitly 2.0s
            thread.start()
        elif action == 'left':
            # Turn left for the specified duration (defaults to 3s based on motor_duration)
            # Intended: Spin left. Actual with current wiring: Likely Left Backward only (pivot left)
            thread = threading.Thread(target=run_single_motor_action_for_duration, args=(turn_left, motor_duration))
            thread.start()
        elif action == 'right':
            # Turn right for the specified duration (defaults to 3s based on motor_duration)
            # Intended: Spin right. Actual with current wiring: Likely no movement
            thread = threading.Thread(target=run_single_motor_action_for_duration, args=(turn_right, motor_duration))
            thread.start()
        elif action == 'stop':
            # Stop action is immediate and doesn't need a thread for duration
            stop()
        elif action == 'thumb up':
             print("Received Thumb up action: Spinning in place for 5 seconds...") # Added print for clarity
             # Spin in place for 5 seconds (Intended: R=F, L=B).
             # Actual with current wiring: R=F, L=F(Nada) -> Only R=F -> Pivot right. (Matches your observation)
             thread = threading.Thread(target=spin_in_place_for_duration, args=(5.0,)) # <-- Call the function defined above
             thread.start()
        elif action == 'thumb down':
             print("Received Thumb down action: Turning 180 then moving forward...") # Added print for clarity
             # Turn ~180 (1.2s) then move forward (2s)
             # Intended turn: Spin. Actual turn with current wiring: R=F, L=B -> Spin! (Matches 'up' behavior)
             # Intended move forward: Both wheels forward. Actual move with current wiring: R=F, L=B -> Spin!
             # Actual thumb down behavior with current wiring: Spin (R=F, L=B) for 1.2s -> Stop -> Spin (R=F, L=B) for 2s.
             turn_duration_180 = 1.2 # Estimated duration for ~180 degree spin
             move_duration_slow = 2.0 # Duration for forward movement
             thread = threading.Thread(target=turn_180_then_move, args=(turn_duration_180, move_duration_slow)) # <-- Call the function defined above
             thread.start()

        # For motor commands executed in a thread, return success immediately
        # The actual motor action runs in the background
        return jsonify({"status": "success", "source": "motor_control", "action": action, "duration": motor_duration if action in ['left', 'right'] else (2.0 if action in ['up', 'down'] else (5.0 if action == 'thumb up' else (f"turn:{turn_duration_180}s, move:{move_duration_slow}s" if action == 'thumb down' else 0))) }), 200 # Adjusted duration in response


    # --- Handle Other Vision-related Actions (unchanged) ---
    elif action in ['alert', 'record', 'log', 'notify', 'light_on', 'buzzer_on', 'none']:
        if action == "alert":
            handle_alert(objects, timestamp)
        elif action == "record":
            handle_record(objects, timestamp)
        elif action == "log":
            handle_log(objects, timestamp)
        elif action == "notify":
            handle_notify(objects, timestamp)
        elif action == "light_on":
            handle_light_on(objects, timestamp)
        elif action == "buzzer_on":
            handle_buzzer_on(objects, timestamp)
        elif action == "none":
            handle_none(objects, timestamp)

        return jsonify({"status": "success", "source": "vision_event", "action": action, "objects": objects}), 200

    else:
        # Handle unknown actions
        handle_unknown(action, objects, timestamp)
        return jsonify({"status": "error", "message": f"Unknown action '{action}'"}), 400

# --- Start Flask App ---

if __name__ == "__main__":
    # Only start Flask server if GPIO initialization was successful
    if gpio_initialized:
        try:
            print(f"GPIO initialized successfully, starting Flask server listening on http://{HOST_IP}:{HOST_PORT}")
            # Use threaded=True to allow Flask to handle multiple requests concurrently
            # debug=True is for development purposes (remove in production)
            # use_reloader=False is added to prevent double execution on file changes,
            # which can cause issues with GPIO initialization being run twice.
            app.run(host=HOST_IP, port=HOST_PORT, debug=True, threaded=True, use_reloader=False)
        except Exception as e:
            print(f"Error occurred while running Flask server: {e}", file=sys.stderr)
        finally:
            # Ensure GPIO resources are cleaned up when Flask exits or an error occurs
            print("Cleaning up GPIO resources...")
            if h is not None and h >= 0:
                 # Stop all motors to ensure pins return to low state before freeing
                 stop() # Attempt to stop motors
                 # Free all claimed GPIO pins
                 for pin in [Motor_R1_Pin, Motor_R2_Pin, Motor_L1_Pin, Motor_L2_Pin]:
                      try:
                           lgpio.gpio_free(h, pin)
                           print(f"Freed GPIO pin BCM {pin}.")
                      except Exception as e:
                           print(f"Error freeing GPIO pin BCM {pin}: {e}", file=sys.stderr)
                 # Close the GPIO chip controller
                 lgpio.gpiochip_close(h)
                 print("GPIO controller closed.")
            else:
                 print("GPIO controller was not successfully opened, no resources to free.")
    else:
        print("GPIO initialization failed. Flask server will not start.")
        # The script would have already exited via sys.exit(1) if initialization failed.
        # This print is more of a fallback message.
