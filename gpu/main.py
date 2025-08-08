import flet as ft
import os
import sys
import platform
from threading import Thread
import datetime
import json

# Default configuration
DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 2000,
    "number_of_vehicles": 30,
    "number_of_walkers": 10,
    "safe": False,
    "filterv": "vehicle.*",
    "generationv": "All",
    "filterw": "walker.pedestrian.*",
    "generationw": "2",
    "tm_port": 8000,
    "asynch": False,
    "hybrid": False,
    "seed": None,
    "seedw": 0,
    "car_lights_on": False,
    "hero": False,
    "respawn": False,
    "no_rendering": False,
    "max_speed": 30.0,
    "vehicle_selection": [
        "vehicle.tesla.cybertruck",
        "vehicle.audi.tt",
        "vehicle.mercedes.coupe_2020",
        "vehicle.jeep.wrangler_rubicon",
        "vehicle.lincoln.mkz_2020"
    ]
}

# Available vehicle blueprints (can be expanded)
AVAILABLE_VEHICLES = [
    "vehicle.tesla.cybertruck",
    "vehicle.audi.tt",
    "vehicle.mercedes.coupe_2020",
    "vehicle.jeep.wrangler_rubicon",
    "vehicle.lincoln.mkz_2020",
    "vehicle.nissan.micra",
    "vehicle.ford.mustang",
    "vehicle.micro.microlino",
    "vehicle.carlamotors.carlacola",
    "vehicle.nissan.patrol",
    "vehicle.ford.crown",
    "vehicle.volkswagen.t2"
]

def main(page: ft.Page):
    page.title = "CARLA Traffic Generator"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 1000
    page.window_height = 800
    page.window_resizable = True
    page.scroll = ft.ScrollMode.AUTO

    # Load or initialize configuration
    try:
        with open("traffic_config.json", "r") as f:
            config = json.load(f)
    except:
        config = DEFAULT_CONFIG.copy()

    # Initialize logs storage
    execution_logs = []

    # UI Controls
    # ===========
    
    # Main configuration controls
    host = ft.TextField(label="Host", value=config["host"], width=200)
    port = ft.TextField(label="Port", value=str(config["port"]), width=100, input_filter=ft.InputFilter(allow=True, regex_string=r"[0-9]"))
    num_vehicles = ft.Slider(
        label="Number of Vehicles", 
        min=1, 
        max=100, 
        divisions=99, 
        value=config["number_of_vehicles"],
        width=400
    )
    num_walkers = ft.Slider(
        label="Number of Walkers", 
        min=0, 
        max=50, 
        divisions=50, 
        value=config["number_of_walkers"],
        width=400
    )
    max_speed = ft.Slider(
        label="Max Speed (m/s)", 
        min=1, 
        max=50, 
        divisions=49, 
        value=config["max_speed"],
        width=400
    )
    
    # Checkboxes for boolean options
    safe_spawn = ft.Checkbox(label="Safe Spawning", value=config["safe"])
    car_lights = ft.Checkbox(label="Car Lights On", value=config["car_lights_on"])
    hero_mode = ft.Checkbox(label="Hero Vehicle", value=config["hero"])
    respawn = ft.Checkbox(label="Respawn Vehicles", value=config["respawn"])
    no_rendering = ft.Checkbox(label="No Rendering", value=config["no_rendering"])
    async_mode = ft.Checkbox(label="Asynchronous Mode", value=config["asynch"])
    hybrid_mode = ft.Checkbox(label="Hybrid Mode", value=config["hybrid"])
    
    # Dropdowns for generation filters
    vehicle_gen = ft.Dropdown(
        label="Vehicle Generation",
        options=[
            ft.dropdown.Option("All"),
            ft.dropdown.Option("1"),
            ft.dropdown.Option("2"),
            ft.dropdown.Option("3"),
        ],
        value=config["generationv"],
        width=150
    )
    
    walker_gen = ft.Dropdown(
        label="Walker Generation",
        options=[
            ft.dropdown.Option("1"),
            ft.dropdown.Option("2"),
            ft.dropdown.Option("All"),
        ],
        value=config["generationw"],
        width=150
    )
    
    # Vehicle selection - multi-select
    vehicle_selection = ft.Column(
        [ft.Text("Select Vehicles to Include:")] +
        [ft.Checkbox(label=v, value=v in config["vehicle_selection"]) for v in AVAILABLE_VEHICLES],
        scroll=ft.ScrollMode.AUTO,
        height=200,
        width=400
    )
    
    # Status controls
    status_text = ft.Text("Ready to generate traffic", size=16, color=ft.Colors.GREY_600)
    progress_ring = ft.ProgressRing(width=20, height=20, visible=False)
    
    # Logs table
    logs_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Time")),
            ft.DataColumn(ft.Text("Action")),
            ft.DataColumn(ft.Text("Status")),
            ft.DataColumn(ft.Text("Details")),
        ],
        rows=[],
        width=900
    )
    
    # Logs container with scroll
    logs_container = ft.Column(
        [ft.Text("Execution Logs", size=18, weight=ft.FontWeight.BOLD), logs_table],
        scroll=ft.ScrollMode.ALWAYS,
        height=300,
    )
    
    # Helper functions
    # ===============
    
    def add_log_entry(action, status, details):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        execution_logs.insert(0, {
            "time": timestamp,
            "action": action,
            "status": status,
            "details": details
        })
        
        # Update table (show only last 20 entries)
        logs_table.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(log["time"])),
                    ft.DataCell(ft.Text(log["action"])),
                    ft.DataCell(ft.Text(log["status"], color=
                        ft.Colors.GREEN if log["status"] == "Success" else 
                        ft.Colors.ORANGE if log["status"] == "Warning" else 
                        ft.Colors.RED
                    )),
                    ft.DataCell(ft.Text(log["details"])),
                ]
            ) for log in execution_logs[:20]
        ]
        page.update()
    
    def save_config():
        config["host"] = host.value
        config["port"] = int(port.value)
        config["number_of_vehicles"] = int(num_vehicles.value)
        config["number_of_walkers"] = int(num_walkers.value)
        config["safe"] = safe_spawn.value
        config["filterv"] = "vehicle.*"  # Overridden by vehicle selection
        config["generationv"] = vehicle_gen.value
        config["filterw"] = "walker.pedestrian.*"
        config["generationw"] = walker_gen.value
        config["tm_port"] = 8000
        config["asynch"] = async_mode.value
        config["hybrid"] = hybrid_mode.value
        config["seed"] = None  # Can be added as a field if needed
        config["seedw"] = 0
        config["car_lights_on"] = car_lights.value
        config["hero"] = hero_mode.value
        config["respawn"] = respawn.value
        config["no_rendering"] = no_rendering.value
        config["max_speed"] = float(max_speed.value)
        config["vehicle_selection"] = [v for v in AVAILABLE_VEHICLES 
                                     if any(c.value and c.label == v for c in vehicle_selection.controls[1:])]
        
        with open("traffic_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        add_log_entry("Configuration", "Success", "Settings saved successfully")
    
    def generate_traffic():
        # Save current configuration first
        save_config()
        
        # Path to the traffic generator script
        script_path = os.path.join(os.path.dirname(__file__), "generate_traffic.py")
        
        if not os.path.exists(script_path):
            status_text.value = "Error: generate_traffic.py not found!"
            status_text.color = ft.Colors.RED
            progress_ring.visible = False
            add_log_entry("Traffic Generation", "Failed", "Script not found")
            page.update()
            return

        try:
            status_text.value = "Generating traffic..."
            status_text.color = ft.Colors.BLUE
            progress_ring.visible = True
            page.update()
            
            # Prepare command with all arguments
            cmd_args = [
                f"--host {config['host']}",
                f"--port {config['port']}",
                f"--number-of-vehicles {config['number_of_vehicles']}",
                f"--number-of-walkers {config['number_of_walkers']}",
                f"--max-speed {config['max_speed']}",
                f"--generationv {config['generationv']}",
                f"--generationw {config['generationw']}",
                f"--tm-port {config['tm_port']}",
            ]
            
            # Add boolean flags
            if config["safe"]: cmd_args.append("--safe")
            if config["car_lights_on"]: cmd_args.append("--car-lights-on")
            if config["hero"]: cmd_args.append("--hero")
            if config["respawn"]: cmd_args.append("--respawn")
            if config["no_rendering"]: cmd_args.append("--no-rendering")
            if config["asynch"]: cmd_args.append("--asynch")
            if config["hybrid"]: cmd_args.append("--hybrid")
            
            # Add vehicle selection (custom implementation in the script)
            for vehicle in config["vehicle_selection"]:
                cmd_args.append(f"--include-vehicle {vehicle}")
            
            command = f'python "{script_path}" {" ".join(cmd_args)}'
            
            # Platform-specific terminal commands
            if platform.system() == "Windows":
                full_command = f'start cmd /k "{command}"'
            elif platform.system() == "Darwin":  # macOS
                full_command = f'''osascript -e 'tell application "Terminal" to do script "{command}"\''''
            else:  # Linux
                full_command = f'x-terminal-emulator -e "{command}"'
            
            def execute_command():
                try:
                    os.system(full_command)
                    status_text.value = "Traffic generation started!"
                    status_text.color = ft.Colors.GREEN
                    add_log_entry("Traffic Generation", "Success", f"Started with {config['number_of_vehicles']} vehicles, {config['number_of_walkers']} walkers")
                except Exception as e:
                    status_text.value = f"Error: {str(e)}"
                    status_text.color = ft.Colors.RED
                    add_log_entry("Traffic Generation", "Failed", str(e))
                finally:
                    progress_ring.visible = False
                    page.update()
            
            Thread(target=execute_command, daemon=True).start()
            
        except Exception as e:
            status_text.value = f"Error: {str(e)}"
            status_text.color = ft.Colors.RED
            progress_ring.visible = False
            add_log_entry("Traffic Generation", "Failed", str(e))
            page.update()
    
    # Create action buttons
    save_button = ft.ElevatedButton(
        text="Save Configuration",
        icon=ft.Icons.SAVE,
        on_click=lambda e: save_config(),
        width=200
    )
    
    generate_button = ft.ElevatedButton(
        text="Generate Traffic",
        icon=ft.Icons.PLAY_ARROW,
        on_click=lambda e: generate_traffic(),
        width=200,
        style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE)
    )
    
    # Layout the page
    # ==============
    
    # Network settings
    network_settings = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.Text("Network Settings", size=16, weight=ft.FontWeight.BOLD),
                ft.Row([host, port]),
            ]),
            padding=10
        )
    )
    
    # Vehicle settings
    vehicle_settings = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.Text("Vehicle Settings", size=16, weight=ft.FontWeight.BOLD),
                num_vehicles,
                max_speed,
                ft.Row([vehicle_gen, safe_spawn, car_lights, hero_mode]),
                vehicle_selection
            ]),
            padding=10
        )
    )
    
    # Walker settings
    walker_settings = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.Text("Walker Settings", size=16, weight=ft.FontWeight.BOLD),
                num_walkers,
                ft.Row([walker_gen])
            ]),
            padding=10
        )
    )
    
    # Simulation settings
    sim_settings = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.Text("Simulation Settings", size=16, weight=ft.FontWeight.BOLD),
                ft.Row([async_mode, hybrid_mode, respawn, no_rendering]),
            ]),
            padding=10
        )
    )
    
    # Action buttons
    action_buttons = ft.Row(
        [save_button, generate_button],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=20
    )
    
    # Status row
    status_row = ft.Row(
        [progress_ring, status_text],
        alignment=ft.MainAxisAlignment.CENTER,
    )
    
    # Main layout
    page.add(
        ft.Column(
            [
                ft.Row(
                    [
                        ft.Image(src="https://carla.org/img/logo.png", width=100),
                        ft.Text("CARLA Traffic Generator", size=24, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Divider(height=20),
                network_settings,
                vehicle_settings,
                walker_settings,
                sim_settings,
                ft.Divider(height=20),
                action_buttons,
                status_row,
                ft.Divider(height=20),
                logs_container,
            ],
            spacing=10,
            expand=True,
        )
    )

if __name__ == "__main__":
    ft.app(target=main)