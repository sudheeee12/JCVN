import os
import subprocess
import time
import sys
import shutil
import re
import argparse
import atexit
import socket

try:
    import psutil
    from colorama import Fore, Style, init
    from dotenv import set_key, find_dotenv
except ImportError:
    print("Required packages not found. Please run: pip install -r requirements.txt")
    sys.exit(1)

# --- Configuration ---
class Config:
    FLASK_APP_FILE = "app.py"
    FLASK_HOST = "127.0.0.1"
    FLASK_PORT = 5000
    HARDHAT_DIR = "blockchain"
    HARDHAT_HOST = "127.0.0.1"
    HARDHAT_PORT = 8545
    PID_FILE = ".run.pids"
    TEMPLATE_DIR = "stored_templates"
    HARDHAT_ACCOUNT_0_PK = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
    RPC_URL = f"http://{HARDHAT_HOST}:{HARDHAT_PORT}"

# Initialize Colorama for cross-platform colored output
init(autoreset=True)

# --- Process Management ---
running_processes = []

def cleanup_processes():
    """Ensure all child processes are terminated on exit."""
    if not running_processes:
        return
    log_step("Shutting down all services...", "[--]")
    for proc_info in reversed(running_processes):
        try:
            parent = psutil.Process(proc_info['pid'])
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            parent.wait(timeout=5)
            log_success(f"{proc_info['name']} shut down.")
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            log_error(f"Process {proc_info['name']} did not terminate gracefully, killing it.")
            parent.kill()
        except Exception as e:
            log_error(f"Error during shutdown of {proc_info['name']}: {e}")
    if os.path.exists(Config.PID_FILE):
        os.remove(Config.PID_FILE)

atexit.register(cleanup_processes)

# --- Logging Helpers ---
def log_step(message, icon="[>>]"):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{icon} {message}{Style.RESET_ALL}")

def log_success(message):
    print(f"{Fore.GREEN}[✓] {message}{Style.RESET_ALL}")

def log_error(message):
    print(f"{Fore.RED}{Style.BRIGHT}[!] ERROR: {message}{Style.RESET_ALL}")

def log_info(message):
    print(f"{Fore.YELLOW}    - {message}{Style.RESET_ALL}")

# --- Command Helpers ---
def get_command(cmd):
    """Get the correct command for the OS (for npm/npx on Windows)."""
    return f"{cmd}.cmd" if sys.platform == "win32" else cmd

def check_command_exists(cmd, install_info):
    """Check if a command exists and print info if it doesn't."""
    if not shutil.which(cmd):
        log_error(f"Command '{cmd}' not found. Please install it.")
        log_info(install_info)
        return False
    return True

def wait_for_port(port, host=Config.HARDHAT_HOST, timeout=30.0):
    """Wait until a port is open before continuing."""
    log_info(f"Waiting for service on {host}:{port}...")
    start_time = time.perf_counter()
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                break
        except OSError:
            time.sleep(0.5)
            if time.perf_counter() - start_time >= timeout:
                log_error(f"Port {port} not open after {timeout} seconds.")
                raise TimeoutError()
    log_success(f"Service on port {port} is ready.")

# --- Main Handlers ---
def handle_check():
    """Check for system dependencies and install python/npm packages."""
    log_step("Checking system and package dependencies...", "[?]")
    all_ok = True
    all_ok &= check_command_exists("node", "Install Node.js from https://nodejs.org/")
    all_ok &= check_command_exists(get_command("npm"), "Install npm (usually comes with Node.js)")
    all_ok &= check_command_exists("tesseract", "Install Tesseract OCR (e.g., 'sudo apt install tesseract-ocr' or 'brew install tesseract')")
    if not all_ok:
        sys.exit(1)
    log_success("All system dependencies are installed.")
    try:
        log_info("Installing/updating Python packages from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True, capture_output=True)
        log_success("Python packages are up to date.")
    except subprocess.CalledProcessError as e:
        log_error("Failed to install Python packages.")
        log_info(e.stderr.decode())
        sys.exit(1)
    try:
        log_info("Installing/updating Node.js packages...")
        subprocess.run([get_command("npm"), "install"], cwd=Config.HARDHAT_DIR, check=True, capture_output=True, shell=(sys.platform == "win32"))
        log_success("Node.js packages are up to date.")
    except subprocess.CalledProcessError as e:
        log_error("Failed to install Node.js packages.")
        log_info(e.stderr.decode())
        sys.exit(1)
    log_success("Environment is ready.")

def handle_start():
    """Start Hardhat node, deploy contract, and run Flask app."""
    handle_check()
    log_step("Starting local Hardhat blockchain...")
    try:
        hardhat_process = subprocess.Popen([get_command("npx"), "hardhat", "node", "--hostname", Config.HARDHAT_HOST], cwd=Config.HARDHAT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=(sys.platform == "win32"))
        running_processes.append({'name': 'Hardhat Node', 'pid': hardhat_process.pid})
        wait_for_port(Config.HARDHAT_PORT)
    except (FileNotFoundError, TimeoutError, subprocess.SubprocessError) as e:
        log_error(f"Failed to start Hardhat node: {e}")
        sys.exit(1)
    log_step("Deploying smart contract...")
    try:
        deploy_command = [get_command("npx"), "hardhat", "run", "scripts/deploy.js", "--network", "localhost"]
        deploy_result = subprocess.run(deploy_command, cwd=Config.HARDHAT_DIR, capture_output=True, text=True, check=True, shell=(sys.platform == "win32"))
        output = deploy_result.stdout
        match = re.search(r"StoreMerkleRoot deployed to: (0x[a-fA-F0-9]{40})", output)
        if not match:
            raise RuntimeError("Could not parse contract address from deployment script output.")
        contract_address = match.group(1)
        log_success(f"Contract deployed successfully at: {contract_address}")
    except (subprocess.CalledProcessError, RuntimeError) as e:
        log_error(f"Failed to deploy contract: {e}")
        sys.exit(1)
    log_step("Updating .env file with new contract address...", "[*]")
    try:
        dotenv_path = find_dotenv()
        if not dotenv_path:
            log_info("No .env file found, creating a new one.")
            dotenv_path = ".env"
            with open(dotenv_path, 'w') as f:
                f.write(f"RPC_URL={Config.RPC_URL}\n")
                f.write(f"SIGNER_PRIVATE_KEY={Config.HARDHAT_ACCOUNT_0_PK}\n")
        set_key(dotenv_path, "CONTRACT_ADDRESS", contract_address)
        log_success(f"Successfully updated CONTRACT_ADDRESS in {os.path.basename(dotenv_path)}")
    except Exception as e:
        log_error(f"Fatal: Could not write to .env file: {e}")
        sys.exit(1)
    log_step("Starting Flask web application...")
    try:
        env = os.environ.copy()
        env["CONTRACT_ADDRESS"] = contract_address
        env["RPC_URL"] = Config.RPC_URL
        env["SIGNER_PRIVATE_KEY"] = Config.HARDHAT_ACCOUNT_0_PK
        flask_process = subprocess.Popen([sys.executable, Config.FLASK_APP_FILE], env=env)
        running_processes.append({'name': 'Flask App', 'pid': flask_process.pid})
        wait_for_port(Config.FLASK_PORT, host=Config.FLASK_HOST)
    except (TimeoutError, subprocess.SubprocessError) as e:
        log_error(f"Failed to start Flask app: {e}")
        sys.exit(1)
    with open(Config.PID_FILE, "w") as f:
        for proc in running_processes:
            f.write(f"{proc['name']}:{proc['pid']}\n")
    log_step(f"All services are running. Access the application at http://{Config.FLASK_HOST}:{Config.FLASK_PORT}", "[COMPLETE]")
    print(f"{Fore.YELLOW}Press Ctrl+C in this terminal to stop all services.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

def handle_reset():
    """Stop running processes and clean up generated files."""
    log_step("Resetting the entire environment...", "[~]")
    if os.path.exists(Config.PID_FILE):
        log_info("Stopping running processes found in .run.pids...")
        with open(Config.PID_FILE, "r") as f:
            lines = f.readlines()
        for line in lines:
            try:
                name, pid_str = line.strip().split(":")
                pid = int(pid_str)
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    cmdline = " ".join(process.cmdline()).lower()
                    if "hardhat" in cmdline or "flask" in cmdline or "python" in cmdline and Config.FLASK_APP_FILE in cmdline:
                        log_info(f"Terminating {name} (PID: {pid})...")
                        process.terminate()
                    else:
                        log_error(f"PID {pid} does not appear to be a demo process. Skipping for safety.")
                else:
                    log_info(f"Process {name} (PID: {pid}) is not running.")
            except (ValueError, psutil.NoSuchProcess):
                continue
            except Exception as e:
                log_error(f"Could not stop process from line '{line.strip()}': {e}")
        os.remove(Config.PID_FILE)
    else:
        log_info("No running processes found (PID file missing).")
    dirs_to_remove = [Config.TEMPLATE_DIR, os.path.join(Config.HARDHAT_DIR, "artifacts"), os.path.join(Config.HARDHAT_DIR, "cache")]
    for d in dirs_to_remove:
        if os.path.isdir(d):
            shutil.rmtree(d)
            log_info(f"Removed directory: {d}")
    if os.path.exists(Config.PID_FILE):
        os.remove(Config.PID_FILE)
    log_success("Environment has been reset.")

def main():
    parser = argparse.ArgumentParser(description="Management script for the IntelliCert OCR Demo.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Check system dependencies and install packages.")
    subparsers.add_parser("start", help="Start the blockchain and Flask application.")
    subparsers.add_parser("reset", help="Stop all services and clean the environment.")
    args = parser.parse_args()
    if args.command == "check":
        handle_check()
    elif args.command == "start":
        handle_start()
    elif args.command == "reset":
        handle_reset()

if __name__ == "__main__":
    main()