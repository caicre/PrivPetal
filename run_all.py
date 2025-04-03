import subprocess
import sys
import time
from datetime import datetime

def run_script(script_name):
    print(f"\n[{datetime.now()}] Starting {script_name}")
    try:
        subprocess.run([sys.executable, script_name], check=True)
        print(f"[{datetime.now()}] Successfully completed {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] Error running {script_name}: {e}")
        return False
    except Exception as e:
        print(f"[{datetime.now()}] Unexpected error running {script_name}: {e}")
        return False

def main():
    scripts = [
        "script_instarcart.py",
        "script.py",
        "script_TPC-H.py"
    ]
    
    print(f"[{datetime.now()}] Starting sequential script execution")
    
    for script in scripts:
        success = run_script(script)
        if not success:
            print(f"[{datetime.now()}] Execution stopped due to error in {script}")
            sys.exit(1)
        time.sleep(1)
    
    print(f"[{datetime.now()}] All scripts completed successfully")

if __name__ == "__main__":
    main()
