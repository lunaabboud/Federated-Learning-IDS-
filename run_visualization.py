# run_visualization.py - Updated version
import subprocess
import os
import time
import sys
from datetime import datetime

def main():
    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Define log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/fl_training_{timestamp}.log"
    
    print(f"Federated Learning IDS Visualization Tool")
    print(f"=======================================")
    print(f"Log file: {log_file}")
    
    # Get the current Python interpreter path
    python_executable = sys.executable
    
    try:
        # Run FL system and capture logs
        print("\nStarting Federated Learning system...")
        process = subprocess.Popen(
            f"docker-compose up | {python_executable} capture_logs.py --output {log_file}",
            shell=True
        )
        
        # Wait for process to complete
        process.wait()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        subprocess.run(f"{python_executable} visualize.py --log {log_file} --output visualizations", shell=True)
        
        print("\nDone! Check the 'visualizations' directory for results.")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted. Stopping...")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
