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
    page.title = "SmartEye"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 1100
    page.window_height = 850
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
    
    # Traffic Generation Controls
    def update_vehicle_label(e):
        num_vehicles.label = f"{int(num_vehicles.value)} vehicles"
        page.update()
    
    def update_walker_label(e):
        num_walkers.label = f"{int(num_walkers.value)} walkers"
        page.update()
    
    def update_speed_label(e):
        max_speed.label = f"{max_speed.value:.1f} m/s"
        page.update()
    
    num_vehicles = ft.Slider(
        # label="Number of Vehicles", 
        min=1, 
        max=100, 
        divisions=99, 
        value=config["number_of_vehicles"],
        width=400,
        label=f"{config['number_of_vehicles']} vehicles",
        on_change=update_vehicle_label
    )
    
    num_walkers = ft.Slider(
        # label="Number of Walkers", 
        min=0, 
        max=50, 
        divisions=50, 
        value=config["number_of_walkers"],
        width=400,
        label=f"{config['number_of_walkers']} walkers",
        on_change=update_walker_label
    )
    
    max_speed = ft.Slider(
        # label="Max Speed (m/s)", 
        min=1, 
        max=50, 
        divisions=49, 
        value=config["max_speed"],
        width=400,
        label=f"{config['max_speed']} m/s",
        on_change=update_speed_label
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
            ft.DataColumn(ft.Text("Time", width=100)),
            ft.DataColumn(ft.Text("Action", width=150)),
            ft.DataColumn(ft.Text("Status", width=100)),
            ft.DataColumn(ft.Text("Details")),
        ],
        rows=[],
        width=1000,
        column_spacing=20
    )
    
    # Logs container
    logs_container = ft.Column(
        [
            ft.Row(
                [
                    ft.Text("Execution Logs", size=18, weight=ft.FontWeight.BOLD),
                    ft.ElevatedButton(
                        "Clear Logs",
                        icon=ft.Icons.CLEAR,
                        on_click=lambda e: clear_logs(),
                        width=150
                    )
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN
            ),
            ft.Container(
                logs_table,
                border=ft.border.all(1, ft.Colors.GREY_400),
                border_radius=5,
                padding=10,
                height=500,
                # scroll=ft.ScrollMode.ALWAYS
            )
        ],
        spacing=10
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
        
        # Update table (show only last 50 entries)
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
                    ft.DataCell(ft.Text(log["details"], selectable=True)),
                ]
            ) for log in execution_logs[:50]
        ]
        page.update()
    
    def clear_logs():
        execution_logs.clear()
        logs_table.rows = []
        add_log_entry("Logs", "Info", "Logs cleared")
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
                    # f"--town {config['town']}",
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

    def start_manual_control():
        # Get manual control configuration
        manual_host = manual_host_field.value
        manual_port = manual_port_field.value
        manual_res = manual_res_field.value
        manual_role = manual_role_field.value
        manual_autopilot = manual_autopilot_check.value
        manual_debug = manual_debug_check.value
        manual_carsim = manual_carsim_check.value
        manual_filter = manual_filter_field.value
        manual_gamma = manual_gamma_field.value
        manual_vehicle = manual_vehicle_dropdown.value
        
        # Update manual control status
        manual_status_text = manual_status
        manual_log_text = manual_log
        
        if not carla_connection.connected:
            manual_status.value = "Not connected to CARLA server!"
            manual_status.color = ft.Colors.RED
            manual_log.value = "Please connect to CARLA server first"
            page.update()
            return
        
        manual_status.value = "Starting manual control..."
        manual_status.color = ft.Colors.BLUE
        manual_log.value = f"Starting manual control with {manual_vehicle}..."
        page.update()
        
        def manual_control_thread():
            try:
                # Prepare command arguments for manual_control_carsim.py
                cmd_args = [
                    f"--host {manual_host}",
                    f"--port {manual_port}",
                    f"--res {manual_res}",
                    f"--rolename {manual_role}",
                    f"--filter {manual_vehicle}",
                    f"--gamma {manual_gamma}",
                ]
                
                # Add boolean flags
                if manual_autopilot: cmd_args.append("--autopilot")
                if manual_debug: cmd_args.append("--verbose")
                
                # Build command
                command = f'python "{os.path.join(os.path.dirname(__file__), "manual_control_carsim.py")}" {" ".join(cmd_args)}'
                
                # Platform-specific terminal commands
                if platform.system() == "Windows":
                    full_command = f'start cmd /k "{command}"'
                elif platform.system() == "Darwin":  # macOS
                    full_command = f'''osascript -e 'tell application "Terminal" to do script "{command}"\''''
                else:  # Linux
                    full_command = f'x-terminal-emulator -e "{command}"'
                
                os.system(full_command)
                
                manual_status.value = "Manual control started!"
                manual_status.color = ft.Colors.GREEN
                manual_log.value = f"Manual control started with {manual_vehicle}"
                
                add_log_entry("Manual Control", "Success", 
                            f"Started manual control with {manual_vehicle}")
                
            except Exception as e:
                manual_status.value = "Failed to start manual control"
                manual_status.color = ft.Colors.RED
                manual_log.value = f"Error: {str(e)}"
                add_log_entry("Manual Control", "Failed", str(e))
            finally:
                page.update()
        
        Thread(target=manual_control_thread, daemon=True).start()
    
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
        content=ft.Container(
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
                ),
                ft.Row([progress_ring, status_text], alignment=ft.MainAxisAlignment.CENTER)
            ]),
            padding=10
        )
    )
    
    # Traffic Generation Tab
    traffic_tab = ft.Tab(
        text="Traffic Generation",
        content=ft.Container(
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
            ]),
            padding=10
        )
    )
    
    # Logs Tab
    logs_tab = ft.Tab(
        text="Logs",
        content=ft.Container(
            content=logs_container,
            padding=10
        )
    )
    
    # Manual Control Tab - Define controls as variables
    manual_status = ft.Text("Disconnected", color=ft.Colors.RED, weight=ft.FontWeight.BOLD)
    manual_log = ft.Text("Ready to start manual control", size=14)
    
    manual_vehicle_dropdown = ft.Dropdown(
        label="Select Vehicle",
        options=[ft.dropdown.Option(v) for v in AVAILABLE_VEHICLES],
        value=AVAILABLE_VEHICLES[0],
        width=300
    )
    
    manual_host_field = ft.TextField(
        label="Host",
        value="127.0.0.1",
        width=150
    )
    
    manual_port_field = ft.TextField(
        label="Port",
        value="2000",
        width=100,
        input_filter=ft.NumbersOnlyInputFilter()
    )
    
    manual_res_field = ft.TextField(
        label="Resolution",
        value="1280x720",
        width=120
    )
    
    manual_role_field = ft.TextField(
        label="Role Name",
        value="hero",
        width=100
    )
    
    manual_autopilot_check = ft.Checkbox(
        label="Enable Autopilot",
        value=False
    )
    
    manual_debug_check = ft.Checkbox(
        label="Enable Debug Mode",
        value=False
    )
    
    manual_carsim_check = ft.Checkbox(
        label="Use Carsim",
        value=False
    )
    
    manual_filter_field = ft.TextField(
        label="Filter Pattern",
        value="vehicle.*",
        width=200
    )
    
    manual_gamma_field = ft.TextField(
        label="Gamma Correction",
        value="2.2",
        width=100
    )
    
    manual_tab = ft.Tab(
        text="Manual Control",
        content=ft.Container(
            content=ft.Column([
                ft.Text("Manual Vehicle Control", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                
                # Connection Status
                ft.Row([
                    ft.Text("CARLA Server:", size=16),
                    manual_status
                ]),
                
                # Vehicle Selection
                ft.Row([
                    manual_vehicle_dropdown,
                    ft.ElevatedButton(
                        text="Start Manual Control",
                        icon=ft.Icons.PLAY_ARROW,
                        on_click=lambda e: start_manual_control(),
                        width=200,
                        style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE)
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                
                ft.Divider(),
                
                # Configuration Options
                ft.Text("Configuration Options", size=16, weight=ft.FontWeight.BOLD),
                
                ft.Row([
                    manual_host_field,
                    manual_port_field,
                    manual_res_field,
                    manual_role_field
                ]),
                
                ft.Row([
                    manual_autopilot_check,
                    manual_debug_check,
                    manual_carsim_check
                ]),
                
                ft.Row([
                    manual_filter_field,
                    manual_gamma_field
                ]),
                
                ft.Divider(),
                
                # Control Instructions
                ft.Text("Keyboard Controls:", size=16, weight=ft.FontWeight.BOLD),
                ft.Column([
                    ft.Text("W/S: Throttle/Brake", size=14),
                    ft.Text("A/D: Steer Left/Right", size=14),
                    ft.Text("Q: Toggle Reverse", size=14),
                    ft.Text("Space: Handbrake", size=14),
                    ft.Text("P: Toggle Autopilot", size=14),
                    ft.Text("ESC: Quit", size=14),
                    ft.Text("H: Show Help", size=14),
                ]),
                
                ft.Divider(),
                
                # Status and Logs
                ft.Text("Manual Control Status", size=16, weight=ft.FontWeight.BOLD),
                ft.Column([
                    manual_log
                ], height=100, scroll=ft.ScrollMode.AUTO),
                
            ], scroll=ft.ScrollMode.AUTO),
            padding=20
        )
    )

    # Detection Tab - Define controls as variables
    detection_status = ft.Text("Ready to start detection", size=14)
    detection_log = ft.Text("Select vehicle to detect", size=14)
    
    # Load car model classes from JSON for detection dropdown
    try:
        with open("gpu/car_model_classes.json", "r") as f:
            detection_vehicle_options = json.load(f)
    except Exception as e:
        detection_vehicle_options = AVAILABLE_VEHICLES
        add_log_entry("Error", "Failed to load car model classes JSON", str(e))

    detection_vehicle_dropdown = ft.Dropdown(
        label="Select Vehicle to Detect",
        options=[ft.dropdown.Option(v) for v in detection_vehicle_options],
        value=detection_vehicle_options[0] if detection_vehicle_options else None,
        width=300
    )
    
    # Detection Logs Tab - New tab for viewing detection results
    detection_logs_data = []
    detection_images_list = []
    
    def load_detection_logs():
        """Load vehicle detection logs from JSON file"""
        try:
            with open('gpu/detection/vehicle_detections.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            add_log_entry("Error", "Failed to load detection logs", str(e))
            return []

    def refresh_detection_logs():
        """Refresh the detection logs display"""
        detection_logs_data.clear()
        detection_logs_data.extend(load_detection_logs())
        
        # Clear existing rows
        detection_logs_table.rows.clear()
        
        # Add new rows
        for detection in detection_logs_data:
            camera = detection.get('camera_label', 'Unknown')
            vehicle_class = detection.get('predicted_class', 'Unknown')
            probability = f"{detection.get('probability', 0):.2%}"
            timestamp = detection.get('timestamp', '')
            
            # Format timestamp
            if timestamp:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass
            
            detection_logs_table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(camera)),
                        ft.DataCell(ft.Text(vehicle_class)),
                        ft.DataCell(ft.Text(probability)),
                        ft.DataCell(ft.Text(timestamp)),
                        ft.DataCell(
                            ft.Row([
                                ft.IconButton(
                                    icon=ft.Icons.IMAGE,
                                    tooltip="View Images",
                                    on_click=lambda e, d=detection: show_detection_images(d)
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.INFO,
                                    tooltip="Details",
                                    on_click=lambda e, d=detection: show_detection_details(d)
                                )
                            ])
                        )
                    ]
                )
            )
        
        detection_logs_count.value = f"Total detections: {len(detection_logs_data)}"
        page.update()

    def show_detection_images(detection):
        """Show images for a specific detection"""
        crop_images = detection.get('crop_image', [])
        if not crop_images:
            detection_image_info.value = "No images available for this detection"
            detection_images_list.clear()
            page.update()
            return
        
        detection_images_list.clear()
        detection_images_list.extend(crop_images)
        
        # Show first image
        if detection_images_list:
            show_image(0)
        
        page.update()

    def show_image(index):
        """Display a specific image from the detection"""
        if index < 0 or index >= len(detection_images_list):
            return
        
        image_data = detection_images_list[index]
        image_path = image_data.get('image_src', '')
        
        if image_path and os.path.exists(image_path):
            try:
                # Load and display image
                from PIL import Image
                img = Image.open(image_path)
                
                # Convert to bytes for display
                import io
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                detection_image_display.src_base64 = base64.b64encode(img_byte_arr.read()).decode()
                detection_image_display.src = None  # Clear any previous src
                
                probability = image_data.get('probability', 0)
                detection_image_info.value = (
                    f"Image: {os.path.basename(image_path)}\n"
                    f"Confidence: {probability:.2%}\n"
                    f"Path: {image_path}"
                )
                
            except Exception as e:
                detection_image_info.value = f"Error loading image: {e}"
        else:
            detection_image_info.value = f"Image not found: {image_path}"

    def show_detection_details(detection):
        """Show detailed information about a detection"""
        details = json.dumps(detection, indent=2)
        detection_details_text.value = details
        page.update()

    # Detection logs table
    detection_logs_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Camera", width=100)),
            ft.DataColumn(ft.Text("Vehicle Class", width=200)),
            ft.DataColumn(ft.Text("Probability", width=100)),
            ft.DataColumn(ft.Text("Timestamp", width=150)),
            ft.DataColumn(ft.Text("Actions", width=100)),
        ],
        rows=[],
        width=800,
        column_spacing=10
    )

    detection_logs_count = ft.Text("Total detections: 0", size=14)
    detection_image_display = ft.Image(width=400, height=300, fit=ft.ImageFit.CONTAIN)
    detection_image_info = ft.Text("Select a detection to view images", size=12)
    detection_details_text = ft.Text("", size=12, selectable=True)

    detection_logs_tab = ft.Tab(
        text="Detection Logs",
        content=ft.Container(
            content=ft.Column([
                ft.Text("Vehicle Detection Logs", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                
                # Controls
                ft.Row([
                    ft.ElevatedButton(
                        text="Refresh Logs",
                        icon=ft.Icons.REFRESH,
                        on_click=lambda e: refresh_detection_logs(),
                        width=150
                    ),
                    detection_logs_count
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                
                # Logs table
                ft.Container(
                    detection_logs_table,
                    border=ft.border.all(1, ft.Colors.GREY_400),
                    border_radius=5,
                    padding=10,
                    height=400,
                ),
                
                ft.Divider(),
                
                # Image display section
                ft.Text("Detection Images", size=16, weight=ft.FontWeight.BOLD),
                ft.Row([
                    ft.Column([
                        detection_image_display,
                        detection_image_info
                    ], width=450),
                    
                    ft.Column([
                        ft.Text("Detection Details", size=14, weight=ft.FontWeight.BOLD),
                        ft.Container(
                            detection_details_text,
                            border=ft.border.all(1, ft.Colors.GREY_400),
                            border_radius=5,
                            padding=10,
                            height=300,
                            width=400,
                        )
                    ], width=450)
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
                
            ], scroll=ft.ScrollMode.AUTO),
            padding=20
        )
    )
    
    detection_tab = ft.Tab(
        text="Detection",
        content=ft.Container(
            content=ft.Column([
                ft.Text("Vehicle Detection System", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                
                # Vehicle Selection
                ft.Row([
                    detection_vehicle_dropdown,
                    ft.ElevatedButton(
                        text="Start Detection",
                        icon=ft.Icons.SEARCH,
                        on_click=lambda e: start_detection(),
                        width=200,
                        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE)
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                
                ft.Divider(),
                
                # Detection Information
                ft.Text("Detection System", size=16, weight=ft.FontWeight.BOLD),
                ft.Column([
                    ft.Text("This system uses YOLO and custom car model classifier", size=14),
                    ft.Text("to detect specific vehicle models in CARLA", size=14),
                    ft.Text("environment with camera feeds.", size=14),
                ]),
                
                ft.Divider(),
                
                # Status and Logs
                ft.Text("Detection Status", size=16, weight=ft.FontWeight.BOLD),
                ft.Column([
                    detection_status,
                    detection_log
                ], height=100, scroll=ft.ScrollMode.AUTO),
                
            ], scroll=ft.ScrollMode.AUTO),
            padding=20
        )
    )
    
    # Function to start detection
    def start_detection():
        detection_vehicle = detection_vehicle_dropdown.value
        detection_status.value = "Starting detection..."
        detection_status.color = ft.Colors.BLUE
        detection_log.value = f"Starting detection for {detection_vehicle}..."
        page.update()
        
        def detection_thread():
            try:
                command = f'D:\ANAS\SmartEye\python310\Scripts\python.exe "{os.path.join(os.path.dirname(__file__), "self_model.py")}" --car_to_detect {detection_vehicle}'
                
                if platform.system() == "Windows":
                    full_command = f'start cmd /k "{command}"'
                elif platform.system() == "Darwin":  # macOS
                    full_command = f'''osascript -e 'tell application "Terminal" to do script "{command}"\''''
                else:  # Linux
                    full_command = f'x-terminal-emulator -e "{command}"'
                
                os.system(full_command)
                
                detection_status.value = "Detection started!"
                detection_status.color = ft.Colors.GREEN
                detection_log.value = f"Detection started for {detection_vehicle}"
                
                add_log_entry("Detection", "Success", f"Started detection for {detection_vehicle}")
            except Exception as e:
                detection_status.value = "Failed to start detection"
                detection_status.color = ft.Colors.RED
                detection_log.value = f"Error: {str(e)}"
                add_log_entry("Detection", "Failed", str(e))
            finally:
                page.update()
        
        Thread(target=detection_thread, daemon=True).start()
    
    # Import the new detection logs tab
    from detection_logs_tab import DetectionLogsTab
    
    # Create detection logs tab instance
    detection_logs_tab_instance = DetectionLogsTab()
    car_tracking_logs_tab = detection_logs_tab_instance.build()
    
    # Main layout with tabs
    tabs = ft.Tabs(
        tabs=[connection_tab, traffic_tab, logs_tab, manual_tab, detection_tab, car_tracking_logs_tab],
        expand=True,
        animation_duration=300
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
                tabs
            ],
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