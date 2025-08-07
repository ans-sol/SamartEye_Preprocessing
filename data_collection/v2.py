#!/usr/bin/env python

import glob
import os
import sys
import math
import random
import time
import threading
import argparse
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Constants
RESNET_IMAGE_SIZE = (224, 224)  # Standard size for ResNet
YOLO_CONFIDENCE = 0.7  # Minimum confidence for detection

def setup_yolo():
    """Initialize YOLOv10 model"""
    try:
        # Install with: pip install ultralytics
        from ultralytics import YOLO
        model = YOLO('yolov10n.pt')  # or yolov10s/m/l/x
        return model
    except Exception as e:
        print(f"Error loading YOLOv10: {e}")
        return None

yolo_model = setup_yolo()

def get_transform(vehicle_location, angle, d=6.0, z_offset=0.0, vehicle_size=1.0):
    """Calculate camera transform with dynamic distance based on vehicle size"""
    a = math.radians(angle)
    size_factor = max(1.0, vehicle_size * 1.2)  # More aggressive size scaling
    adjusted_d = d + z_offset * size_factor
    location = carla.Location(
        adjusted_d * math.cos(a),
        adjusted_d * math.sin(a),
        2.0 + z_offset
    ) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ImageCapture:
    def __init__(self, output_root, vehicle_blueprint):
        self.vehicle_type = vehicle_blueprint.id.replace('.', '_')
        self.color_name = self.get_color_name(vehicle_blueprint)
        self.output_dir = os.path.join(
            output_root,
            self.vehicle_type,
            self.color_name
        )
        ensure_dir(self.output_dir)
        
        self.image_count = 0
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.vehicle_size = 1.0
        self.resize_transform = transforms.Compose([
            transforms.Resize(RESNET_IMAGE_SIZE),
            transforms.Lambda(lambda x: x.convert('RGB'))
        ])

    def get_color_name(self, blueprint):
        """Get color name from vehicle blueprint"""
        if blueprint.has_attribute('color'):
            color = blueprint.get_attribute('color')
            return color.split('/')[-1].split('.')[0]
        return 'default'

    def detect_and_crop(self, image):
        """Process image with YOLOv10 and prepare for ResNet"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]  # RGBA to BGR
        
        # YOLOv10 detection
        results = yolo_model(array) if yolo_model else None
        best_box = None
        
        if results:
            for result in results:
                for box in result.boxes:
                    if int(box.cls) in [2, 5, 7] and box.conf > YOLO_CONFIDENCE:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        if not best_box or (x2-x1)*(y2-y1) > (best_box[2]-best_box[0])*(best_box[3]-best_box[1]):
                            best_box = (x1, y1, x2, y2)
                            self.vehicle_size = ((x2-x1)/image.width + (y2-y1)/image.height)/2
        
        pil_image = Image.fromarray(array[:, :, ::-1])  # Convert to RGB
        
        if best_box:
            x1, y1, x2, y2 = best_box
            # Add 15% margin but keep within image bounds
            margin_w = int((x2 - x1) * 0.15)
            margin_h = int((y2 - y1) * 0.15)
            x1 = max(0, x1 - margin_w)
            y1 = max(0, y1 - margin_h)
            x2 = min(image.width, x2 + margin_w)
            y2 = min(image.height, y2 + margin_h)
            cropped = pil_image.crop((x1, y1, x2, y2))
        else:
            # Center crop fallback
            size = min(image.width, image.height)
            left = (image.width - size) // 2
            top = (image.height - size) // 2
            cropped = pil_image.crop((left, top, left + size, top + size))
        
        # Resize to ResNet input size and ensure RGB
        return self.resize_transform(cropped), bool(best_box)

    def save_image(self, image):
        with self.lock:
            try:
                processed_img, detected = self.detect_and_crop(image)
                filename = os.path.join(
                    self.output_dir,
                    f"{self.vehicle_type}_{self.color_name}_{self.image_count:04d}.png"
                )
                processed_img.save(filename)
                print(f"Saved {'detected' if detected else 'fallback'} image: {filename}")
                self.image_count += 1
                self.ready.set()
            except Exception as e:
                print(f"Error saving image: {e}")
                self.ready.set()

def clear_all_vehicles(world):
    """Destroy all vehicles and wait for them to be removed"""
    vehicles = world.get_actors().filter('vehicle.*')
    print(f"Destroying {len(vehicles)} existing vehicles...")
    for vehicle in vehicles:
        vehicle.destroy()
    
    # Wait for vehicles to be destroyed
    start_time = time.time()
    while time.time() - start_time < 5.0:
        remaining = len(world.get_actors().filter('vehicle.*'))
        if remaining == 0:
            break
        time.sleep(0.1)
    else:
        print("Warning: Some vehicles may not have been destroyed properly")

def spawn_vehicle_with_retry(world, blueprint, transform, max_retries=5):
    """Attempt to spawn a vehicle with retries and collision avoidance"""
    for attempt in range(max_retries):
        try:
            # Try to spawn at a slightly different location each attempt
            adjusted_transform = transform
            if attempt > 0:
                offset = carla.Location(
                    x=random.uniform(-1.0, 1.0),
                    y=random.uniform(-1.0, 1.0),
                    z=0
                )
                adjusted_transform.location += offset
            
            vehicle = world.spawn_actor(blueprint, adjusted_transform)
            print(f"Successfully spawned {blueprint.id} on attempt {attempt + 1}")
            return vehicle
        except RuntimeError as e:
            if "Spawn failed because of collision" in str(e):
                print(f"Spawn collision detected for {blueprint.id}, retrying... (Attempt {attempt + 1})")
                time.sleep(1.0)
            else:
                print(f"Error spawning vehicle: {e}")
                raise
    
    raise RuntimeError(f"Failed to spawn vehicle {blueprint.id} after {max_retries} attempts")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vehicle Image Capture Script')
    parser.add_argument('--rotations', type=int, default=5, help='Number of 360° rotations to perform')
    parser.add_argument('--images-per-rotation', type=int, default=36, help='Number of images per 360° rotation')
    parser.add_argument('--z-increment', type=float, default=0.5, help='Z-axis increment after each rotation')
    parser.add_argument('--initial-z', type=float, default=0.2, help='Initial Z offset for camera (very low position)')
    parser.add_argument('--base-z', type=float, default=0.2, help='Base Z coordinate for vehicle spawning')
    args = parser.parse_args()

    client = carla.Client('localhost', 2000)
    client.set_timeout(12.0)
    world = client.get_world()

    # Clear all existing vehicles
    clear_all_vehicles(world)

    # Only process these specific vehicle types
    target_vehicles = [
        'vehicle.jeep.wrangler_rubicon',
        # 'vehicle.mercedes.coupe_2020',
        # 'vehicle.nissan.micra',
        # 'vehicle.dodge.charger_police',
        # 'vehicle.ford.mustang'
    ]
    
    vehicle_blueprints = [
        bp for bp in world.get_blueprint_library().filter('vehicle.*')
        if bp.id in target_vehicles
    ]

    # Fixed spawn location for consistency
    base_location = carla.Location(x=-44.1, y=21.4, z=args.base_z)  # Very low spawn position
    base_rotation = carla.Rotation(yaw=-45.0)

    for blueprint in vehicle_blueprints:
        # Clear any remaining vehicles before spawning new one
        clear_all_vehicles(world)
        
        print(f"\nPreparing to capture {args.rotations} rotations ({args.rotations * args.images_per_rotation} images) for vehicle: {blueprint.id}")
        
        # Create output directory for this vehicle type
        output_dir = os.path.join('vehicle_images', blueprint.id.replace('.', '_'))
        ensure_dir(output_dir)

        try:
            # Spawn the vehicle with retry mechanism
            transform = carla.Transform(base_location, base_rotation)
            vehicle = spawn_vehicle_with_retry(world, blueprint, transform)
            
            # Wait a moment to let the vehicle settle
            time.sleep(1.0)

            # Spawn camera sensor
            blueprint_library = world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')

            image_capture = ImageCapture(output_dir, blueprint.id.replace('.', '_'))

            # Spawn camera attached to spectator
            camera_transform = get_transform(vehicle.get_location(), 0, z_offset=args.initial_z, vehicle_size=image_capture.vehicle_size)
            camera = world.spawn_actor(camera_bp, camera_transform)
            camera.listen(image_capture.save_image)

            # Calculate angle step based on images per rotation
            angle_step = 360.0 / args.images_per_rotation
            current_z_offset = args.initial_z

            for rotation in range(args.rotations):
                print(f"\nStarting rotation {rotation + 1}/{args.rotations} at Z-offset {current_z_offset:.2f}m")
                
                for frame in range(args.images_per_rotation):
                    angle = frame * angle_step
                    vehicle_loc = vehicle.get_location()
                    camera.set_transform(get_transform(vehicle_loc, angle - 90, 
                                                   z_offset=current_z_offset,
                                                   vehicle_size=image_capture.vehicle_size))
                    image_capture.ready.clear()
                    
                    # Wait for image to be saved
                    if not image_capture.ready.wait(timeout=2.0):
                        print(f"Warning: Timeout waiting for image {image_capture.image_count}")

                # Increment Z after each complete rotation
                current_z_offset += args.z_increment

            camera.stop()
            camera.destroy()

        except Exception as e:
            print(f"Error processing vehicle {blueprint.id}: {e}")
        finally:
            # Ensure vehicle is destroyed
            if 'vehicle' in locals() and vehicle.is_alive:
                vehicle.destroy()
                time.sleep(3.0)  # Give time for destruction to complete

if __name__ == '__main__':
    main()