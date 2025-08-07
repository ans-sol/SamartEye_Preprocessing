import carla

def clear_all_vehicles_and_reset_view():
    try:
        # Connect to CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Get all actors and filter vehicles
        vehicles = [a for a in world.get_actors() if 'vehicle' in a.type_id.lower()]
        
        # Destroy vehicles
        for vehicle in vehicles:
            vehicle.destroy()
        
        # Reset spectator camera
        spectator = world.get_spectator()
        default_transform = carla.Transform(
            carla.Location(x=0, y=0, z=50),  # 50m above origin
            carla.Rotation(pitch=-90)       # Looking straight down
        )
        spectator.set_transform(default_transform)
        
        print(f"Removed {len(vehicles)} vehicles and reset spectator view")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    clear_all_vehicles_and_reset_view()