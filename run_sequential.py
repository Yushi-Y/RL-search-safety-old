#!/usr/bin/env python3
"""
Script to run multiple inference scripts sequentially
"""
import subprocess
import sys
import os

def run_script(script_path, cuda_device=1):
    """Run a single script with CUDA_VISIBLE_DEVICES set"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    
    print(f"Running: {script_path}")
    print(f"CUDA_VISIBLE_DEVICES={cuda_device}")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            env=env,
            check=True,
            capture_output=False  # Show output in real-time
        )
        print(f"✓ {script_path} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {script_path} failed with exit code {e.returncode}")
        return False

def main():
    # Define the scripts to run sequentially
    scripts = [
        "infer_qwen14b_ppo_scripts/infer_search.py",
        "infer_qwen14b_ppo_scripts/infer_search_base.py",  # Replace with your second script
    ]
    
    print("Starting sequential script execution...")
    print(f"Will run {len(scripts)} scripts")
    print("=" * 60)
    
    success_count = 0
    for i, script in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] Starting {script}")
        
        if run_script(script):
            success_count += 1
        else:
            print(f"Script {script} failed. Stopping execution.")
            break
    
    print("\n" + "=" * 60)
    print(f"Execution complete: {success_count}/{len(scripts)} scripts succeeded")
    
    if success_count == len(scripts):
        print("✓ All scripts completed successfully!")
        sys.exit(0)
    else:
        print("✗ Some scripts failed")
        sys.exit(1)

if __name__ == "__main__":
    main()