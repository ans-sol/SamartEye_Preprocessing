import flet as ft
import os
import sys
import platform
from threading import Thread
import datetime
import json
import carla

# Default configuration
DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 2000,
    "town": "Town02",
    "number_of_vehicles": 30,
    "number_of_walkers": 10,
    "safe": False,
    "generationv": "All",
    "generationw": "2",
    "tm_port": 8000,
    "asynch": False,
    "hybrid": False,
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

# Available vehicle blueprints
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

# Available towns
AVAILABLE_TOWNS = [
    "Town01", "Town02", "Town03", "Town04", 
    "Town05", "Town06", "Town07", "Town10HD"
]

class CarlaConnection:
    def __init__(self):
        self.client = None
        self.world = None
        self.connected = False
    
    def connect(self, host, port, timeout=10.0):
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)
            self.world = self.client.get_world()
            self.connected = True
            return True, "Connection successful"
        except Exception as e:
            self.connected = False
            return False, f"Connection failed: {str(e)}"
    
    def load_town(self, town_name):
        if not self.connected:
            return False, "Not connected to CARLA server"
        try:
            self.world = self.client.load_world(town_name)
            return True, f"Loaded {town_name} successfully"
        except Exception as e:
            return False, f"Failed to load town: {str(e)}"
    
    def disconnect(self):
        if self.client:
            self.client = None
        self.connected = False

def main(page: ft.Page):
    page.title = "CARLA Control Center"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 1000
    page.window_height = 800
    page.window_resizable = True
    page.scroll = ft.ScrollMode.AUTO

    # CARLA connection handler
    carla_connection = CarlaConnection()
    
    # Load or initialize configuration
    try:
        with open("carla_config.json", "r") as f:
            config = json.load(f)
    except:
        config = DEFAULT_CONFIG.copy()

    # Initialize logs storage
    execution_logs = []

    # UI Controls
    # ===========
    
    # Connection Status Indicator
    connection_status = ft.Text(
        "Disconnected", 
        color=ft.Colors.RED,
        weight=ft.FontWeight.BOLD
    )
    
    # Connection Controls
    host = ft.TextField(label="Host", value=config["host"], width=200)
    port = ft.TextField(label="Port", value=str(config["port"]), width=100, input_filter=ft.NumbersOnlyInputFilter())
    
    connect_button = ft.ElevatedButton(
        text="Connect",
        icon=ft.Icons.LINK,
        on_click=lambda e: connect_to_carla(),
        width=150
    )
    
    disconnect_button = ft.ElevatedButton(
        text="Disconnect",
        icon=ft.Icons.LINK_OFF,
        on_click=lambda e: disconnect_from_carla(),
        width=150,
        disabled=True
    )
    
    # Town Selection
    town_dropdown = ft.Dropdown(
        label="Select Town",
        options=[ft.dropdown.Option(t) for t in AVAILABLE_TOWNS],
        value=config["town"],
        width=200,
        disabled=True
    )
    
    load_town_button = ft.ElevatedButton(
        text="Load Town",
        icon=ft.Icons.LOCATION_CITY,
        on_click=lambda e: load_town(),
        width=150,
        disabled=True
    )
    
    # Traffic Generation Controls (in a separate tab)
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
    
    # Vehicle selection
    vehicle_selection = ft.Column(
        [ft.Text("Select Vehicles to Include:")] +
        [ft.Checkbox(label=v, value=v in config["vehicle_selection"]) for v in AVAILABLE_VEHICLES],
        scroll=ft.ScrollMode.AUTO,
        height=200,
        width=400
    )
    
    # Status controls
    status_text = ft.Text("Ready", size=16, color=ft.Colors.GREY_600)
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
    
    # Logs container
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
    
    def update_connection_status(connected):
        if connected:
            connection_status.value = "Connected"
            connection_status.color = ft.Colors.GREEN
            connect_button.disabled = True
            disconnect_button.disabled = False
            town_dropdown.disabled = False
            load_town_button.disabled = False
        else:
            connection_status.value = "Disconnected"
            connection_status.color = ft.Colors.RED
            connect_button.disabled = False
            disconnect_button.disabled = True
            town_dropdown.disabled = True
            load_town_button.disabled = True
        
        page.update()
    
    def connect_to_carla():
        status_text.value = "Connecting to CARLA..."
        status_text.color = ft.Colors.BLUE
        progress_ring.visible = True
        page.update()
        
        def connect_thread():
            try:
                success, message = carla_connection.connect(host.value, int(port.value))
                if success:
                    add_log_entry("CARLA Connection", "Success", message)
                    status_text.value = message
                    status_text.color = ft.Colors.GREEN
                    update_connection_status(True)
                else:
                    add_log_entry("CARLA Connection", "Failed", message)
                    status_text.value = message
                    status_text.color = ft.Colors.RED
            except Exception as e:
                add_log_entry("CARLA Connection", "Error", str(e))
                status_text.value = f"Error: {str(e)}"
                status_text.color = ft.Colors.RED
            finally:
                progress_ring.visible = False
                page.update()
        
        Thread(target=connect_thread, daemon=True).start()
    
    def disconnect_from_carla():
        carla_connection.disconnect()
        update_connection_status(False)
        add_log_entry("CARLA Connection", "Info", "Disconnected from CARLA server")
        status_text.value = "Disconnected from CARLA"
        status_text.color = ft.Colors.GREY_600
    
    def load_town():
        if not town_dropdown.value:
            status_text.value = "Please select a town first!"
            status_text.color = ft.Colors.RED
            page.update()
            return
        
        status_text.value = f"Loading {town_dropdown.value}..."
        status_text.color = ft.Colors.BLUE
        progress_ring.visible = True
        page.update()
        
        def load_thread():
            try:
                success, message = carla_connection.load_town(town_dropdown.value)
                if success:
                    add_log_entry("Town Load", "Success", message)
                    status_text.value = message
                    status_text.color = ft.Colors.GREEN
                    config["town"] = town_dropdown.value
                    save_config()
                else:
                    add_log_entry("Town Load", "Failed", message)
                    status_text.value = message
                    status_text.color = ft.Colors.RED
            except Exception as e:
                add_log_entry("Town Load", "Error", str(e))
                status_text.value = f"Error: {str(e)}"
                status_text.color = ft.Colors.RED
            finally:
                progress_ring.visible = False
                page.update()
        
        Thread(target=load_thread, daemon=True).start()
    
    def save_config():
        config["host"] = host.value
        config["port"] = int(port.value)
        config["town"] = town_dropdown.value
        config["number_of_vehicles"] = int(num_vehicles.value)
        config["number_of_walkers"] = int(num_walkers.value)
        config["safe"] = safe_spawn.value
        config["generationv"] = vehicle_gen.value
        config["generationw"] = walker_gen.value
        config["asynch"] = async_mode.value
        config["hybrid"] = hybrid_mode.value
        config["car_lights_on"] = car_lights.value
        config["hero"] = hero_mode.value
        config["respawn"] = respawn.value
        config["no_rendering"] = no_rendering.value
        config["max_speed"] = float(max_speed.value)
        config["vehicle_selection"] = [v for v in AVAILABLE_VEHICLES 
                                     if any(c.value and c.label == v for c in vehicle_selection.controls[1:])]
        
        with open("carla_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        add_log_entry("Configuration", "Success", "Settings saved successfully")
    
    def generate_traffic():
        if not carla_connection.connected:
            status_text.value = "Not connected to CARLA server!"
            status_text.color = ft.Colors.RED
            page.update()
            return
        
        save_config()
        
        status_text.value = "Generating traffic..."
        status_text.color = ft.Colors.BLUE
        progress_ring.visible = True
        page.update()
        
        def generate_thread():
            try:
                # Prepare command with all arguments
                cmd_args = [
                    f"--host {config['host']}",
                    f"--port {config['port']}",
                    f"--town {config['town']}",
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
                
                # Add vehicle selection
                for vehicle in config["vehicle_selection"]:
                    cmd_args.append(f"--include-vehicle {vehicle}")
                
                command = f'python "{os.path.join(os.path.dirname(__file__), "generate_traffic.py")}" {" ".join(cmd_args)}'
                
                # Platform-specific terminal commands
                if platform.system() == "Windows":
                    full_command = f'start cmd /k "{command}"'
                elif platform.system() == "Darwin":  # macOS
                    full_command = f'''osascript -e 'tell application "Terminal" to do script "{command}"\''''
                else:  # Linux
                    full_command = f'x-terminal-emulator -e "{command}"'
                
                os.system(full_command)
                status_text.value = "Traffic generation started!"
                status_text.color = ft.Colors.GREEN
                add_log_entry("Traffic Generation", "Success", 
                            f"Started with {config['number_of_vehicles']} vehicles, {config['number_of_walkers']} walkers")
            except Exception as e:
                status_text.value = f"Error: {str(e)}"
                status_text.color = ft.Colors.RED
                add_log_entry("Traffic Generation", "Failed", str(e))
            finally:
                progress_ring.visible = False
                page.update()
        
        Thread(target=generate_thread, daemon=True).start()
    
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
    
    # Tab Layout
    # =========
    
    # Connection Tab
    connection_tab = ft.Tab(
        text="Connection",
        content=ft.Column([
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("CARLA Connection", size=18, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            host,
                            port,
                            ft.Column([
                                ft.Row([connect_button, disconnect_button]),
                                ft.Row([connection_status], alignment=ft.MainAxisAlignment.CENTER)
                            ])
                        ]),
                        ft.Divider(),
                        ft.Row([
                            town_dropdown,
                            load_town_button
                        ])
                    ]),
                    padding=20
                )
            )
        ])
    )
    
    # Traffic Generation Tab
    traffic_tab = ft.Tab(
        text="Traffic Generation",
        content=ft.Column([
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("Traffic Settings", size=18, weight=ft.FontWeight.BOLD),
                        num_vehicles,
                        num_walkers,
                        max_speed,
                        ft.Row([vehicle_gen, walker_gen]),
                        ft.Row([safe_spawn, car_lights, hero_mode, respawn]),
                        ft.Row([async_mode, hybrid_mode, no_rendering]),
                        vehicle_selection,
                        ft.Row([save_button, generate_button], alignment=ft.MainAxisAlignment.CENTER)
                    ]),
                    padding=20
                )
            )
        ])
    )
    
    # Manual Control Tab (placeholder for future)
    manual_tab = ft.Tab(
        text="Manual Control",
        content=ft.Column([
            ft.Text("Manual vehicle control will be implemented here", size=16),
            ft.Text("This is a placeholder for future development", style=ft.TextStyle(italic=True))
        ])
    )
    
    # Main layout with tabs
    tabs = ft.Tabs(
        tabs=[connection_tab, traffic_tab, manual_tab],
        expand=True
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
                        ft.Text("CARLA Control Center", size=24, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Divider(height=20),
                tabs,
                status_row,
                ft.Divider(height=20),
                logs_container,
            ],
            spacing=10,
            expand=True,
        )
    )

if __name__ == "__main__":
    # Try to import carla to check if it's available
    try:
        import carla
        ft.app(target=main)
    except ImportError:
        print("CARLA Python API not found. Please make sure it's installed and in your PYTHONPATH.")
        print("On Windows, typically you need to add the .egg file to your path:")
        print("e.g., sys.path.append(r'path\\to\\carla\\PythonAPI\\carla\\dist\\carla-X.X.egg')")