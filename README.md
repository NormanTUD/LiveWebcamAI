LiveWebcamAI - README
=====================

This project connects a browser-based webcam interface with a Stable-Diffusion-like model running on a remote HPC cluster via Slurm (`sbatch`). The architecture uses HTTP for communication between frontend and backend, and a persistent SSH tunnel to forward requests to the compute node.

Directory Structure:
--------------------
backend/
├── main.py              → Backend service that submits Slurm jobs via SSH
├── requirements.txt     → Backend dependencies
├── run.md               → Optional usage instructions
├── run.sbatch           → Slurm batch script to launch the model

frontend/
├── main.py              → Web server (e.g. Flask or FastAPI)
├── requirements.txt     → Frontend dependencies
├── run.sh               → Script to launch frontend server
├── static/              → Static resources (JS, CSS, etc.)
├── templates/           → HTML templates

How it Works:
-------------
1. A Slurm job is launched using `sbatch`, with the image as input.
2. The user opens the website, which accesses their webcam.
3. A still frame is sent to the backend via HTTP.
4. The backend forwards the image to the remote HPC over SSH.
5. The backend waits synchronously for the result.
6. Once the output image is generated, it is returned to the frontend and displayed.

Requirements:
-------------
- Python 3.10+
- SSH access to an HPC cluster with Slurm installed
- Passwordless SSH key authentication (already set up)
- A persistent SSH tunnel must be running for port forwarding
- Webcam access from a modern browser

Setup:
------
1. Frontend:
   `cd frontend`
   `./run.sh`
   # Visit http://localhost:5000 in your browser

2. Backend:
   ssh to HPC
   `cd backend`
   `sbatch run.sbatch`
   # The backend connects to HPC via SSH and submits the job using run.sbatch

3. Set up a persistent SSH tunnel (e.g. in a `tmux` session):
   `ssh -L 9932:localhost:9932 -J jumphost_username@jumphost_server.com,hpc_user@login.hpc_system.de node_user@node_name`

Notes:
------
- The model job is launched synchronously via Slurm; the backend waits for completion.
- The Slurm script (`run.sbatch`) must save the output image in a known location.
- The backend polls or blocks until the result is available, then sends it back.
- The SSH tunnel must remain open during operation.
