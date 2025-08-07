# SmartEye - CARLA Simulation Tools

## Overview
SmartEye is a collection of Python scripts designed for working with the CARLA autonomous driving simulator. These tools provide functionality for traffic generation, vehicle management, camera positioning, and data collection.

## File Descriptions

### Core Scripts

#### `generate_traffic.py`
**Purpose**: Generates realistic traffic scenarios in CARLA simulation
- **Functionality**: Spawns vehicles and pedestrians with configurable parameters
- **Key Features**:
  - Spawn specified number of vehicles and walkers
  - Filter vehicle types using blueprint patterns
  - Safe spawning mode to prevent collisions
  - Traffic manager integration for realistic behavior
  - Configurable max speed and autopilot settings
- **Usage**: `python generate_traffic.py --number-of-vehicles 20 --number-of-walkers 0 --tm-port 8000 --safe`

#### `clear_vehicles.py`
**Purpose**: Utility to clean up all vehicles and reset the simulation view
- **Functionality**: Removes all spawned vehicles and resets spectator camera
- **Key Features**:
  - Connects to CARLA server and identifies all vehicles
  - Destroys all vehicle actors safely
  - Resets spectator camera to default overhead view
  - Error handling for connection issues
- **Usage**: `python clear_vehicles.py`

#### `save_camera_positions.py`
**Purpose**: Interactive tool for saving camera positions and orientations
- **Functionality**: GUI-based camera position picker with Pygame interface
- **Key Features**:
  - Real-time camera position visualization
  - Save current spectator transforms with mouse clicks
  - Load and manage saved camera positions
  - Export positions to JSON format
  - Support for multiple camera setups
- **Usage**: `python save_camera_positions.py`

### Data Collection Scripts

#### `data_collection/` Directory
Contains multiple versions of data collection scripts for training computer vision models:

- **`v1.py` through `v7.py`**: Progressive versions of data collection scripts
- **Features**:
  - Automatic image capture from CARLA cameras
  - YOLO model integration for object detection
  - Dataset generation for training autonomous driving models
  - Support for multiple vehicle types and scenarios

### Supporting Files

#### `gpu_camera.py`
**Purpose**: GPU-accelerated camera processing for real-time image capture
- **Functionality**: High-performance camera feed processing
- **Features**:
  - GPU acceleration for image processing
  - Real-time camera feed handling
  - Integration with machine learning pipelines

#### `manual_control_carsim.py`
**Purpose**: Manual vehicle control interface for testing
- **Functionality**: Keyboard-based vehicle control in simulation
- **Features**:
  - Real-time vehicle control using keyboard inputs
  - Camera view switching
  - Vehicle state monitoring
  - Debug information display

#### `camera_transforms.json`
**Purpose**: Stores saved camera positions and orientations
- **Format**: JSON file containing transform data (location and rotation)
- **Usage**: Used by `save_camera_positions.py` to persist camera setups

### Model Files

#### `yolov8n.pt`
**Purpose**: Pre-trained YOLOv8 nano model for object detection
- **Usage**: Used in data collection scripts for vehicle and pedestrian detection
- **Size**: Optimized for real-time inference with minimal computational overhead

## Quick Start

1. **Start CARLA Server**: Launch CARLA simulator
2. **Generate Traffic**: `python generate_traffic.py --number-of-vehicles 20 --safe`
3. **Clear Traffic**: `python clear_vehicles.py`
4. **Set Camera Positions**: `python save_camera_positions.py`
5. **Collect Data**: Navigate to `data_collection/` and run appropriate version

## Dependencies
- CARLA Python API
- Pygame (for GUI tools)
- PyTorch (for ML models)
- OpenCV (for image processing)

## Usage Examples

### Basic Traffic Generation
```bash
python generate_traffic.py --number-of-vehicles 30 --number-of-walkers 10 --tm-port 8000 --safe
```

### Clean Up Simulation
```bash
python clear_vehicles.py
```

### Interactive Camera Setup
```bash
python save_camera_positions.py
# Use mouse clicks to save positions, 'S' to save, ESC to exit
```

### Data Collection
```bash
cd data_collection
python v7.py  # Latest version with full features
```

## Notes
- Ensure CARLA server is running before executing any scripts
- Default connection settings: localhost:2000
- All scripts include error handling for connection issues
- Camera positions are automatically saved to `camera_transforms.json`
