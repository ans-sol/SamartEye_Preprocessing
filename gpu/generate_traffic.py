#!/usr/bin/env python

import glob
import os
import sys
import time
import argparse

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls
import logging
from numpy import random

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)
    
    if generation.lower() == "all":
        return bps

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def main():
    argparser = argparse.ArgumentParser(description='CARLA Traffic Generator')
    
    # Required arguments
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('--port', type=int, default=2000, help='TCP port to listen to')
    argparser.add_argument('--number-of-vehicles', type=int, default=30, help='Number of vehicles')
    argparser.add_argument('--number-of-walkers', type=int, default=10, help='Number of walkers')
    argparser.add_argument('--max-speed', type=float, default=30.0, help='Maximum speed for vehicles')
    
    # Vehicle selection
    argparser.add_argument('--include-vehicle', action='append', default=[], 
                         help='Specific vehicle models to include (can be used multiple times)')
    
    # Other options
    argparser.add_argument('--safe', action='store_true', help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument('--generationv', default='All', help='Vehicle generation filter')
    argparser.add_argument('--generationw', default='2', help='Walker generation filter')
    argparser.add_argument('--filterw', default='walker.pedestrian.*', help='Walker blueprint filter')
    argparser.add_argument('--tm-port', type=int, default=8000, help='Traffic Manager port')
    argparser.add_argument('--asynch', action='store_true', help='Asynchronous mode execution')
    argparser.add_argument('--hybrid', action='store_true', help='Hybrid mode for Traffic Manager')
    argparser.add_argument('--car-lights-on', action='store_true', help='Enable automatic car light management')
    argparser.add_argument('--hero', action='store_true', help='Set one vehicle as hero')
    argparser.add_argument('--respawn', action='store_true', help='Automatically respawn dormant vehicles')
    argparser.add_argument('--no-rendering', action='store_true', help='Activate no rendering mode')

    args = argparser.parse_args()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    synchronous_master = False
    random.seed(int(time.time()))

    try:
        world = client.load_world('Town02')

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(50.0)
        
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)

        settings = world.get_settings()
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
        else:
            print("Running in asynchronous mode")

        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        # Get blueprints for selected vehicles
        if args.include_vehicle:
            blueprints = []
            for vehicle in args.include_vehicle:
                bp = world.get_blueprint_library().find(vehicle)
                if bp is not None:
                    blueprints.append(bp)
                else:
                    print(f"Warning: Vehicle {vehicle} not found in blueprint library.")
        else:
            blueprints = get_actor_blueprints(world, "vehicle.*", args.generationv)

        if not blueprints:
            raise ValueError("No valid vehicle blueprints found")

        if args.safe:
            blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            logging.warning(f"Requested {args.number_of_vehicles} vehicles, but only {number_of_spawn_points} spawn points available")
            args.number_of_vehicles = number_of_spawn_points

        # Spawn vehicles
        batch = []
        hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            batch.append(carla.command.SpawnActor(blueprint, transform)
                .then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)

        # Spawn walkers
        if args.number_of_walkers > 0:
            walker_bps = get_actor_blueprints(world, args.filterw, args.generationw)
            spawn_points = []
            for i in range(args.number_of_walkers):
                spawn_point = carla.Transform()
                loc = world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

            batch = []
            walker_speed = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(walker_bps)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                speed = walker_bp.get_attribute('speed').recommended_values[1]  # Normal speed
                walker_speed.append(speed)
                batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

            results = client.apply_batch_sync(batch, True)
            walker_speeds = []
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speeds.append(walker_speed[i])

            # Create walker controllers
            batch = []
            walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(walkers_list)):
                batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))

            results = client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
                    all_id.append(walkers_list[i]["con"])
                    all_id.append(walkers_list[i]["id"])

            all_actors = world.get_actors(all_id)
            for i in range(0, len(all_id), 2):
                all_actors[i].start()
                all_actors[i].go_to_location(world.get_random_location_from_navigation())
                all_actors[i].set_max_speed(float(walker_speeds[int(i/2)]))

        print(f"Spawned {len(vehicles_list)} vehicles and {len(walkers_list)} walkers")
        print("Press Ctrl+C to exit")

        # Main loop
        while True:
            if not args.asynch and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    except KeyboardInterrupt:
        print("\nCancelled by user")
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        # Cleanup
        if 'world' in locals():
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

            print(f"\nDestroying {len(vehicles_list)} vehicles")
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

            if 'walkers_list' in locals() and len(walkers_list) > 0:
                print(f"Destroying {len(walkers_list)} walkers")
                for i in range(0, len(all_id), 2):
                    all_actors[i].stop()
                client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

if __name__ == '__main__':
    main()