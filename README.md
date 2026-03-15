# robotics# TRON1 Control Utilities and RAG Setup

This repository contains a small collection of Python utilities for two practical workflows:

1. **TRON1 robot control and recovery** over WebSocket
2. **Recipe RAG index setup** using ChromaDB

The current codebase is mostly centered on **TRON1 teleoperation, recovery, and camera-triggered safety actions**, with one standalone script for building a **recipe retrieval index**.

## Repository Contents

### Robot control / TRON1 tools

- **`tron_client.py`**  
  Lightweight persistent WebSocket client for TRON1. It wraps common robot commands such as stand, walk, stair mode, emergency stop, gait selection, height adjustment, and twist commands.

- **`manual_control_wheeled.py`**  
  Terminal-based manual controller optimized for **wheel-focused driving**. It supports mode switching, acceleration-limited velocity ramping, profiles (`precision`, `normal`, `sport`), gait toggling, height adjustment, camera enable/disable, deadman safety, and recovery helpers.

- **`keyboard_teleop_mujoco.py`**  
  Keyboard teleoperation script for **MuJoCo simulation**. It is intended for simulation-side testing and tuning before deploying control ideas on hardware.

- **`recover_startup_mode.py`**  
  Recovery/probing utility that attempts to move the robot out of **DAMPING** mode and test which stand/walk commands are accepted by the current firmware. Useful when command names differ across firmware versions.

- **`camera_trigger_ml.py`**  
  Camera-triggered action pipeline for safety or reactive behaviors. It supports:
  - a no-dependency brightness heuristic
  - Ultralytics YOLO
  - ONNX Runtime YOLO-style models
  
  When a hazard is detected, the script latches an **emergency stop** and periodically sends zero twist commands for a configurable hold window.

### RAG / retrieval setup

- **`setup_rag.py`**  
  One-time setup script for building a **ChromaDB recipe index**. It can ingest a subset of **Food.com** data when local CSV files are available, or fall back to a bundled sample recipe dataset for demos.

## Features

### TRON1 utilities

- Persistent WebSocket robot client
- Manual wheeled teleoperation from terminal
- MuJoCo keyboard teleop for simulation experiments
- Startup recovery and firmware command probing
- Camera-triggered emergency-stop workflow
- Configurable motion profiles and safety behaviors
- Environment-variable-based configuration for easy deployment

### RAG utility

- Builds a ChromaDB vector index for recipes
- Supports Food.com CSV ingestion with optional rating enrichment
- Falls back to sample recipes for quick demo setup
- Environment-variable configuration for subset size and DB path

## Requirements

### Python

Python **3.10+** is recommended.

### Common dependencies

Install the base dependencies first:

```bash
pip install websocket-client python-dotenv pandas
```

### Optional dependencies by script

#### For `manual_control_wheeled.py`
No extra packages beyond `websocket-client` are required.

#### For `recover_startup_mode.py`
```bash
pip install websocket-client
```

#### For `keyboard_teleop_mujoco.py`
```bash
pip install mujoco glfw numpy
```

> On macOS, MuJoCo viewer is often more reliable when launched with `mjpython` instead of plain `python`.

#### For `camera_trigger_ml.py`
OpenCV backend:
```bash
pip install opencv-python websocket-client
```

Ultralytics detector:
```bash
pip install ultralytics
```

ONNX detector:
```bash
pip install onnxruntime
```

ROS backend requires a working ROS environment with packages such as `rospy`, `sensor_msgs`, and `cv_bridge`.

#### For `setup_rag.py`
In addition to the base dependencies, this script expects the project to contain a retrieval module such as:

```text
modules/rag_retriever.py
```

That module should provide at least:
- `index_recipes(...)`
- `reset_collection(...)`

You will also need Chroma/embedding dependencies required by your retriever implementation.

## Environment Variables

### TRON1 connection

Most robot scripts use:

```bash
export TRON_WS_URL=ws://10.192.1.2:5000
export TRON_ACCID=WF_TRON1A_377
```

### Manual control tuning

`manual_control_wheeled.py` also supports profile overrides such as:

```bash
export TRON_CTRL_PROFILE=normal
export TRON_STEP_VX=0.05
export TRON_MAX_VX=0.45
export TRON_ACCEL_VX=0.80
```

### Camera-triggered actions

Examples:

```bash
export CAMERA_BACKEND=opencv
export CAMERA_DEVICE=0
export MODEL_PROVIDER=none
export HOLD_SECONDS=1.5
export COOLDOWN_SECONDS=0.5
```

For ML detectors:

```bash
export MODEL_PROVIDER=ultralytics
export MODEL_CLASSES=person,car
export MODEL_SCORE_THR=0.5
```

or

```bash
export MODEL_PROVIDER=onnx
export MODEL_PATH=path/to/model.onnx
```

### RAG setup

Examples:

```bash
export RECIPE_SUBSET_SIZE=20000
export INTERACTIONS_CHUNK_SIZE=300000
export CHROMA_DB_PATH=./chroma_db
```

Embedding configuration is delegated to your retriever module, typically through variables like:

```bash
export EMBEDDING_PROVIDER=...
export EMBEDDING_MODEL=...
```

## Data Layout for RAG

`setup_rag.py` expects a local structure like this:

```text
project_root/
├── setup_rag.py
├── data/
│   ├── sample_recipes.json
│   ├── technique_qna.json
│   ├── RAW_recipes.csv
│   └── RAW_interactions.csv
└── modules/
    └── rag_retriever.py
```

### Food.com ingestion

To use Food.com data:

1. Download the dataset from Kaggle.
2. Place the files in `data/`.
3. Run:

```bash
python setup_rag.py
```

If `RAW_recipes.csv` is missing, the script falls back to `sample_recipes.json`.

## Usage

### 1. Manual wheeled control

```bash
python manual_control_wheeled.py --ws-url ws://10.192.1.2:5000 --accid WF_TRON1A_377
```

Useful keys:

- `1` stand
- `2` wheel mode
- `3` walk mode
- `4` stair mode
- `5` sit
- `W/S` forward/backward target
- `A/D` yaw turning
- `Q/E` lateral target
- `P/N/M` precision/normal/sport profile
- `G` toggle gait
- `R/F` height up/down
- `Space` emergency stop
- `Esc` quit

### 2. Recover startup mode

```bash
python recover_startup_mode.py
```

This script:
- connects to the robot
- sends an emergency stop first for safety
- probes candidate stand commands
- probes candidate walk/wheel commands
- prints a suggested environment override if a command works

### 3. Camera-triggered stop logic

```bash
python camera_trigger_ml.py
```

Typical OpenCV demo:

```bash
export CAMERA_BACKEND=opencv
export CAMERA_DEVICE=0
export MODEL_PROVIDER=none
python camera_trigger_ml.py
```

Ultralytics example:

```bash
export MODEL_PROVIDER=ultralytics
export MODEL_CLASSES=person
python camera_trigger_ml.py
```

ONNX example:

```bash
export MODEL_PROVIDER=onnx
export MODEL_PATH=./model.onnx
python camera_trigger_ml.py
```

### 4. MuJoCo keyboard teleop

```bash
python keyboard_teleop_mujoco.py
```

On macOS:

```bash
mjpython keyboard_teleop_mujoco.py
```

### 5. Build the recipe RAG index

```bash
python setup_rag.py
```

## Safety Notes

These scripts can command real robot motion. Before running them on hardware:

- verify network connectivity and the correct `TRON_ACCID`
- keep a physical emergency stop available
- test new behaviors in simulation first when possible
- validate firmware-specific command names before relying on automation
- treat camera-triggered actions as safety assists, not as a full certified safety system

## Suggested Project Structure

If you want to make this repository easier to maintain, consider reorganizing it like this:

```text
.
├── robot/
│   ├── tron_client.py
│   ├── manual_control_wheeled.py
│   ├── keyboard_teleop_mujoco.py
│   ├── recover_startup_mode.py
│   └── camera_trigger_ml.py
├── rag/
│   ├── setup_rag.py
│   ├── data/
│   └── modules/
└── README.md
```

## Troubleshooting

### WebSocket connection fails

- Confirm the robot is reachable on the network.
- Check `TRON_WS_URL` and `TRON_ACCID`.
- Verify the robot firmware exposes the expected WebSocket endpoint.

### Robot remains in DAMPING mode

- Run `recover_startup_mode.py`.
- If recovery still fails, use the official app/remote first, then rerun the probe.

### Camera status does not become `OK`

- Override `CAMERA_ENABLE_COMMANDS` for your firmware.
- Check camera hardware state and robot status telemetry.

### RAG indexing fails

- Verify `modules/rag_retriever.py` exists.
- Check that Chroma and embedding dependencies are installed.
- Confirm `data/sample_recipes.json` or Food.com CSV files are present.

## License

Add your preferred license here, for example MIT, Apache-2.0, or a proprietary internal-use notice.

## Acknowledgments

- MuJoCo for simulation tooling
- ChromaDB and your embedding stack for retrieval infrastructure
- Food.com dataset contributors if using the recipe ingestion workflow
