import time
import math
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

import controlPanel as cp
from droneManager import Drone, PIDController

def test_pid_convergence():
    print("Testing PID Convergence...")
    
    # Initialize Drone
    drone = Drone(0, 0.0, 0.0)
    
    # Set target
    target_x, target_y = 100.0, 50.0
    drone.add_waypoint(target_x, target_y)
    
    print(f"Start Pos: {drone.pos}, Target: ({target_x}, {target_y})")
    
    # Simulation Loop
    dt = 1/60.0
    sim_time = 0.0
    max_time = 10.0 # seconds
    
    history = []
    
    while sim_time < max_time:
        drone.update_physics(dt)
        sim_time += dt
        
        history.append((sim_time, list(drone.pos), list(drone.vel), list(drone.acc)))
        
        # Check if reached
        dist = math.sqrt((drone.pos[0] - target_x)**2 + (drone.pos[1] - target_y)**2)
        if dist < 0.5 and not drone.waypoints:
            print(f"Reached target at t={sim_time:.2f}s")
            print(f"Final Pos: {drone.pos}")
            break
            
    if drone.waypoints: # Still has waypoint
        print(f"FAILED to reach target in {max_time}s. Dist: {dist:.2f}")
    else:
        print("SUCCESS: Target reached.")
        
    return history

if __name__ == "__main__":
    test_pid_convergence()
