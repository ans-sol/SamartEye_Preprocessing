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
MAX_RETRIES = 3  # Max retries for YOLO detection

def setup_yolo():
    """Initialize YOLOv10 model with error handling"""
    try:
        from ultralytics import YOLO
        try:
            model = YOLO('yolov10n.pt')
            print("YOLOv10 model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading YOLOv10 weights: {e}")
            print("Attempting to download weights...")
            try:
                model = YOLO('yolov10n.pt', task='detect')
                return model
            except Exception as e:
                print(f"Failed to download weights: {e}")
                return None
    except ImportError:
        print("ultralytics package not found. Install with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"Unexpected error loading YOLO: {e}")
        return None

yolo_model = setup_yolo()

def get_transform(vehicle_location, angle, d=6.0, z_offset=0.0, vehicle_size=1.0):
    """Calculate camera transform with dynamic distance based on vehicle size"""
    a = math.radians(angle)
    size_factor = max(1.0, vehicle_size * 1.2)
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
            try:
                color = blueprint.get_attribute('color')
                return color.split('/')[-1].split('.')[0].replace(' ', '_')
            except:
                pass
        return 'default'

    def safe_detect(self, image_array, retries=MAX_RETRIES):
        """Safe wrapper for YOLO detection with retries"""
        for attempt in range(retries):
            try:
                results = yolo_model(image_array) if yolo_model else None
                return results
            except Exception as e:
                print(f"Detection attempt {attempt + 1} failed: {e}")
                time.sleep(1)
        return None

    def detect_and_crop(self, image):
        """Process image with YOLOv10 and prepare for ResNet"""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3][:, :, ::-1]  # RGBA to BGR
            
            # YOLOv10 detection with retries
            results = self.safe_detect(array)
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
                # Add margin but keep within bounds
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
            
            return self.resize_transform(cropped), bool(best_box)
        except Exception as e:
            print(f"Error in detect_and_crop: {e}")
            # Return blank image if processing fails
            blank = Image.new('RGB', RESNET_IMAGE_SIZE, (0, 0, 0))
            return blank, False

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
            vehicle = world.spawn_actor(blueprint, transform)
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
    parser = argparse.ArgumentParser(description='Vehicle Image Capture Script')
    parser.add_argument('--rotations', type=int, default=5, help='Number of 360° rotations to perform')
    parser.add_argument('--images-per-rotation', type=int, default=36, help='Number of images per 360° rotation')
    parser.add_argument('--z-increment', type=float, default=0.5, help='Z-axis increment after each rotation')
    parser.add_argument('--initial-z', type=float, default=0.2, help='Initial Z offset for camera')
    parser.add_argument('--base-z', type=float, default=0.2, help='Base Z coordinate for vehicle spawning')
    args = parser.parse_args()

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(12.0)
        world = client.get_world()

        # Clear all existing vehicles
        clear_all_vehicles(world)

        target_vehicles = [
            'vehicle.jeep.wrangler_rubicon',
            'vehicle.mercedes.coupe_2020',
            'vehicle.nissan.micra',
            'vehicle.dodge.charger_police',
            'vehicle.ford.mustang'
        ]
        
        vehicle_blueprints = [
            bp for bp in world.get_blueprint_library().filter('vehicle.*')
            if bp.id in target_vehicles
        ]

        # Fixed spawn location
        base_location = carla.Location(x=-44.1, y=21.4, z=args.base_z)
        base_rotation = carla.Rotation(yaw=-45.0)

        for blueprint in vehicle_blueprints:
            clear_all_vehicles(world)
            print(f"\nProcessing {blueprint.id}...")
            
            try:
                transform = carla.Transform(base_location, base_rotation)
                vehicle = spawn_vehicle_with_retry(world, blueprint, transform)
                time.sleep(1.0)  # Let vehicle settle

                # Setup camera
                camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
                camera_bp.set_attribute('image_size_x', '800')
                camera_bp.set_attribute('image_size_y', '600')
                camera_bp.set_attribute('fov', '90')

                image_capture = ImageCapture('dataset', blueprint)

                # Initial camera position
                camera_transform = get_transform(vehicle.get_location(), 0, z_offset=args.initial_z)
                camera = world.spawn_actor(camera_bp, camera_transform)
                camera.listen(image_capture.save_image)

                angle_step = 360.0 / args.images_per_rotation
                current_z_offset = args.initial_z

                for rotation in range(args.rotations):
                    print(f"Rotation {rotation + 1}/{args.rotations} at Z={current_z_offset:.2f}m")
                    
                    for frame in range(args.images_per_rotation):
                        angle = frame * angle_step
                        vehicle_loc = vehicle.get_location()
                        camera.set_transform(get_transform(
                            vehicle_loc, 
                            angle - 90, 
                            z_offset=current_z_offset,
                            vehicle_size=image_capture.vehicle_size
                        ))
                        image_capture.ready.clear()
                        
                        if not image_capture.ready.wait(timeout=2.0):
                            print(f"Warning: Timeout waiting for image {image_capture.image_count}")

                    current_z_offset += args.z_increment

                camera.stop()
                camera.destroy()
                vehicle.destroy()
                time.sleep(1.0)

            except Exception as e:
                print(f"Error processing vehicle {blueprint.id}: {e}")
                if 'camera' in locals() and camera.is_alive:
                    camera.destroy()
                if 'vehicle' in locals() and vehicle.is_alive:
                    vehicle.destroy()

    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        print("Cleaning up...")
        if 'world' in locals():
            clear_all_vehicles(world)
        print("Done")

if __name__ == '__main__':
    main()