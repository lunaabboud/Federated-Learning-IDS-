# visualize_existing.py - Updated version
import os
import glob
import subprocess
import sys
from datetime import datetime

def main():
    print("Federated Learning IDS - Visualize Existing Logs")
    print("===============================================")
    
    # Find log files
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_files = glob.glob(f"{log_dir}/*.log")
    
    if not log_files:
        print("No log files found in the 'logs' directory.")
        return
    
    # Sort log files by modification time (newest first)
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    print("\nAvailable log files:")
    for i, log_file in enumerate(log_files):
        mod_time = datetime.fromtimestamp(os.path.getmtime(log_file)).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i+1}. {os.path.basename(log_file)} - Last modified: {mod_time}")
    
    # Get user selection
    selection = input("\nEnter the number of the log file to visualize (or press Enter for the most recent): ")
    
    if selection.strip() == "":
        selected_log = log_files[0]
    else:
        try:
            idx = int(selection) - 1
            if idx < 0 or idx >= len(log_files):
                print("Invalid selection. Using the most recent log file.")
                selected_log = log_files[0]
            else:
                selected_log = log_files[idx]
        except ValueError:
            print("Invalid input. Using the most recent log file.")
            selected_log = log_files[0]
    
    print(f"\nVisualizing log file: {selected_log}")
    
    # Get the current Python interpreter path
    python_executable = sys.executable
    
    # Run visualization
    subprocess.run(f"{python_executable} visualize.py --log {selected_log} --output visualizations", shell=True)

if __name__ == "__main__":
    main()
