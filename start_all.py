# start_all.py

"""
Start all DocuBot services
"""
import subprocess
import time
import sys
import os
from pathlib import Path

def start_service(name, command, cwd=None):
    print(f"Starting {name}...")
    try:
        # For Windows, use creationflags to prevent new console window
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            creationflags=creationflags,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        time.sleep(3)  # Give more time to start
        return process
    except Exception as e:
        print(f"Failed to start {name}: {e}")
        return None

def check_service(port):
    """Check if service is running on port"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        return result == 0
    except:
        return False

def find_project_root():
    """Find the project root by looking for specific files"""
    current = Path.cwd()
    
    # Check multiple parent directories
    for depth in range(5):
        test_path = current
        for _ in range(depth):
            test_path = test_path.parent
        
        # Check for project markers
        markers = [
            test_path / "src" / "ui" / "web" / "app.py",
            test_path / "src" / "ui" / "web" / "streamlit_app.py",
            test_path / "requirements.txt",
            test_path / "pyproject.toml"
        ]
        
        if any(marker.exists() for marker in markers):
            return test_path
    
    return None

if __name__ == "__main__":
    print("=" * 50)
    print("DOCUBOT SERVICE MANAGER")
    print("=" * 50)
    
    # Find project root dynamically
    project_root = find_project_root()
    
    if not project_root:
        print("ERROR: Cannot find DocuBot project root.")
        print("Please run this script from within the DocuBot project directory.")
        print("Looking for: src/ui/web/app.py, requirements.txt, or pyproject.toml")
        sys.exit(1)
    
    print(f"Project root: {project_root}")
    print(f"Current directory: {Path.cwd()}")
    
    # Check if we need to change directory
    if Path.cwd() != project_root:
        print(f"Changing to project directory: {project_root}")
        os.chdir(project_root)
    
    # Verify required files exist
    required_files = [
        project_root / "src" / "ui" / "web" / "app.py",
        project_root / "src" / "ui" / "web" / "streamlit_app.py"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"WARNING: Required file not found: {file_path}")
    
    services = [
        {
            "name": "FastAPI", 
            "command": f'"{sys.executable}" -m uvicorn src.ui.web.app:app --host 0.0.0.0 --port 8000 --reload',
            "port": 8000,
            "file": "src/ui/web/app.py"
        },
        {
            "name": "Streamlit", 
            "command": f'"{sys.executable}" -m streamlit run src/ui/web/streamlit_app.py --server.port 8501 --server.headless true',
            "port": 8501,
            "file": "src/ui/web/streamlit_app.py"
        }
    ]
    
    processes = []
    print("\nStarting services...")
    
    for service in services:
        # Check if service file exists
        service_file = project_root / service["file"]
        if not service_file.exists():
            print(f"SKIP {service['name']}: File not found - {service_file}")
            continue
        
        # Check if service is already running
        if check_service(service["port"]):
            print(f"{service['name']} is already running on port {service['port']}")
            continue
            
        print(f"\nStarting {service['name']}...")
        print(f"Command: {service['command']}")
        print(f"Directory: {project_root}")
        
        proc = start_service(service["name"], service["command"], cwd=project_root)
        if proc:
            processes.append((service["name"], proc, service["port"]))
            print(f"{service['name']} started (PID: {proc.pid})")
        else:
            print(f"Failed to start {service['name']}")
    
    print("\n" + "-" * 50)
    print("SERVICE STATUS:")
    print("-" * 50)
    
    # Wait a bit for services to fully start
    time.sleep(2)
    
    all_running = True
    for service in services:
        if check_service(service["port"]):
            print(f"✓ {service['name']}: Running on port {service['port']}")
        else:
            print(f"✗ {service['name']}: Not responding on port {service['port']}")
            all_running = False
    
    print("\n" + "-" * 50)
    print("ACCESS LINKS:")
    print("-" * 50)
    print("FastAPI:     http://localhost:8000")
    print("Streamlit:   http://localhost:8501")
    print("Ollama:      http://localhost:11434")
    print("\n" + "-" * 50)
    print("Press Ctrl+C to stop all services")
    print("-" * 50)
    
    try:
        # Monitor services
        while True:
            time.sleep(5)
            # Check if services are still running
            for name, proc, port in list(processes):
                if proc.poll() is not None:
                    print(f"\n{name} has stopped unexpectedly (exit code: {proc.returncode})")
                    processes.remove((name, proc, port))
            
            if not processes:
                print("\nAll services have stopped")
                break
                
    except KeyboardInterrupt:
        print("\n\nStopping services...")
        for name, proc, _ in processes:
            try:
                print(f"Stopping {name} (PID: {proc.pid})...")
                proc.terminate()
                proc.wait(timeout=5)
                print(f"✓ {name} stopped")
            except subprocess.TimeoutExpired:
                print(f"Force stopping {name}...")
                proc.kill()
            except Exception as e:
                print(f"Error stopping {name}: {e}")
        
        print("\nAll services stopped")