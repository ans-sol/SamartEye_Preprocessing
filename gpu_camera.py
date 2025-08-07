#!/usr/bin/env python
import carla
import pygame
import numpy as np
import json
import math
import cv2
import time
import os
from collections import deque
from ultralytics import YOLO
import torch

# Initialize Pygame
pygame.init()
grid_width, grid_height = 1200, 800
grid_display = pygame.display.set_mode((grid_width, grid_height))
pygame.display.set_caption("CARLA Camera Grid with GPU Vehicle Detection")

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model (will automatically use GPU if available)
model = YOLO('yolov8n.pt').to(device)  # nano version for real-time performance

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Set synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Video recording parameters
RECORDING_WIDTH = 1280
RECORDING_HEIGHT = 720
FPS = 30
DETECTION_THRESHOLD = 0.5  # Confidence threshold for vehicle detection

class CameraRenderObject:
    def __init__(self, width, height, label):
        self.width = width
        self.height = height
        self.label = label
        self.surface = pygame.Surface((width, height))
        self.font = pygame.font.SysFont('Arial', 20)
        self.last_frame = None
        self.vehicle_boxes = []
        self.fps_queue = deque(maxlen=10)
        self.detection_fps_queue = deque(maxlen=10)
        self.last_time = time.time()
        self.last_detection_time = time.time()

    def update(self, image):
        # Process RGB image
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = img.reshape((image.height, image.width, 4))
        img = img[:, :, :3]  # Remove alpha
        img = img[:, :, ::-1]  # BGR to RGB
        self.last_frame = img.copy()
        
        # Calculate FPS
        current_time = time.time()
        self.fps_queue.append(1/(current_time - self.last_time + 1e-6))
        self.last_time = current_time
        
        # Resize for display
        if self.width != image.width or self.height != image.height:
            img = cv2.resize(img, (self.width, self.height))
        self.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

    def detect_vehicles(self):
        if self.last_frame is None:
            return

        start_time = time.time()
        
        # Run YOLO detection
        results = model(self.last_frame, verbose=False, classes=[2, 3, 5, 7])  # Cars, bikes, buses, trucks
        
        # Process detections
        self.vehicle_boxes = []
        scale_x = self.width / self.last_frame.shape[1]
        scale_y = self.height / self.last_frame.shape[0]
        
        for result in results:
            for box in result.boxes:
                if box.conf.item() > DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    self.vehicle_boxes.append((
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int((x2 - x1) * scale_x),
                        int((y2 - y1) * scale_y)
                    ))
        
        # Calculate detection FPS
        self.detection_fps_queue.append(1/(time.time() - start_time + 1e-6))

def setup_cameras():
    # Load camera positions
    json_filename = 'camera_transforms.json'
    try:
        with open(json_filename) as f:
            camera_data = json.load(f)
        print(f"Loaded {len(camera_data)} camera positions")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No camera positions found")
        return [], []

    street_cameras = []
    street_renders = []

    if camera_data:
        # Calculate grid layout
        num_cameras = len(camera_data)
        grid_cols = math.ceil(math.sqrt(num_cameras))
        grid_rows = math.ceil(num_cameras / grid_cols)
        cell_width = grid_width // grid_cols
        cell_height = grid_height // grid_rows
        
        # Camera blueprint
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(RECORDING_WIDTH))
        camera_bp.set_attribute('image_size_y', str(RECORDING_HEIGHT))
        camera_bp.set_attribute('fov', '90')
        
        for i, cam_info in enumerate(camera_data):
            # Create transform
            loc = cam_info["location"]
            rot = cam_info["rotation"]
            transform = carla.Transform(
                carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
                carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
            )
            
            # Create render object
            render = CameraRenderObject(cell_width, cell_height, f"Cam {i+1}")
            street_renders.append(render)
            
            # Spawn camera
            try:
                camera = world.spawn_actor(camera_bp, transform)
                camera.listen(lambda image, idx=i: street_renders[idx].update(image))
                street_cameras.append(camera)
            except Exception as e:
                print(f"Failed to spawn camera {i+1}: {e}")

    return street_cameras, street_renders

def create_video_writers(base_path, num_cameras):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(base_path, exist_ok=True)
    
    writers = []
    for i in range(num_cameras):
        filename = os.path.join(base_path, f"camera_{i+1}_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(
            filename, 
            fourcc, 
            FPS, 
            (RECORDING_WIDTH, RECORDING_HEIGHT),
            isColor=True
        )
        if not writer.isOpened():
            print(f"Failed to initialize video writer for camera {i+1}")
            writer = None
        writers.append(writer)
    return writers

def draw_detections(surface, boxes, fps, detection_fps):
    # Draw bounding boxes
    for box in boxes:
        x, y, w, h = box
        pygame.draw.rect(surface, (255, 0, 0), (x, y, w, h), 2)
    
    # Draw FPS counters
    fps_text = f"Display: {fps:.1f} FPS"
    det_fps_text = f"Detection: {detection_fps:.1f} FPS"
    
    font = pygame.font.SysFont('Arial', 16)
    text_surface = font.render(fps_text, True, (255, 255, 255))
    surface.blit(text_surface, (10, 10))
    
    text_surface = font.render(det_fps_text, True, (255, 255, 255))
    surface.blit(text_surface, (10, 30))

def main():
    street_cameras, street_renders = setup_cameras()
    if not street_cameras:
        return

    clock = pygame.time.Clock()
    running = True
    recording = False
    video_writers = []
    grid_cols = math.ceil(math.sqrt(len(street_renders)))
    detection_interval = 0.1  # Run detection every 100ms (10 FPS)

    try:
        last_detection_time = time.time()
        while running:
            clock.tick(FPS)
            world.tick()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        recording = not recording
                        if recording:
                            print("Starting recording...")
                            video_writers = create_video_writers("recordings", len(street_cameras))
                        else:
                            print("Stopping recording...")
                            for writer in video_writers:
                                if writer is not None:
                                    writer.release()
                            video_writers = []
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            # Run detection periodically
            current_time = time.time()
            if current_time - last_detection_time > detection_interval:
                for render in street_renders:
                    render.detect_vehicles()
                last_detection_time = current_time
            
            # Update display
            grid_display.fill((0, 0, 0))
            
            for i, render in enumerate(street_renders):
                # Draw camera surface
                surface = render.surface.copy()
                
                # Draw detections and FPS
                avg_fps = sum(render.fps_queue)/len(render.fps_queue) if render.fps_queue else 0
                avg_det_fps = sum(render.detection_fps_queue)/len(render.detection_fps_queue) if render.detection_fps_queue else 0
                draw_detections(surface, render.vehicle_boxes, avg_fps, avg_det_fps)
                
                # Draw recording indicator
                if recording:
                    pygame.draw.circle(surface, (255, 0, 0), (render.width - 20, 20), 10)
                    # Write frame to video if writer exists
                    if i < len(video_writers) and video_writers[i] is not None and render.last_frame is not None:
                        frame = cv2.cvtColor(render.last_frame, cv2.COLOR_RGB2BGR)
                        video_writers[i].write(frame)
                
                # Position in grid
                row = i // grid_cols
                col = i % grid_cols
                x_pos = col * render.width
                y_pos = row * render.height
                
                # Draw to display
                grid_display.blit(surface, (x_pos, y_pos))
                label = render.font.render(render.label, True, (255, 255, 255))
                grid_display.blit(label, (x_pos + 10, y_pos + 10))
            
            pygame.display.flip()

    finally:
        # Clean up
        print("Cleaning up...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        for camera in street_cameras:
            if camera.is_alive:
                camera.destroy()
        
        for writer in video_writers:
            if writer is not None and writer.isOpened():
                writer.release()
        
        pygame.quit()

if __name__ == "__main__":
    main()