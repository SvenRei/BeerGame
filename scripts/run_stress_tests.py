import os
import sys
import subprocess

scenarios = {
    "baseline": "env.demand_type=poisson env.jittery_lead_time=False",
    "jitter": "env.demand_type=poisson env.jittery_lead_time=True",
    "black_swan": "env.demand_type=black_swan env.jittery_lead_time=False"
}

# Get the path to your project root (one level up from 'scripts')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prepare the environment with the correct PYTHONPATH
my_env = os.environ.copy()
my_env["PYTHONPATH"] = project_root

print(f"--- Starting Automated Stress Test Suite ---")

for name, cmd_args in scenarios.items():
    print(f"\n[+] Running Scenario: {name.upper()}")
    
    # Use sys.executable to ensure we use the same venv
    cmd = [sys.executable, "-m", "scripts.evaluate"] + cmd_args.split()
    
    result = subprocess.run(cmd, env=my_env, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error in {name}:")
        print(result.stderr)