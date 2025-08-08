import flet as ft
import os
import sys
import platform
from threading import Thread
import datetime
import json

# Vehicle data (could also load from a file/database)
VEHICLES = [
    {"id": "vehicle.mercedes.coupe_2020", "name": "Mercedes Coupe 2020", "type": "Gas"},
    {"id": "vehicle.nissan.micra", "name": "Nissan Micra", "type": "Gas"},
    {"id": "vehicle.ford.mustang", "name": "Ford Mustang", "type": "Gas"},
    {"id": "vehicle.lincoln.mkz_2020", "name": "Lincoln MKZ 2020", "type": "Gas"},
    {"id": "vehicle.dodge.charger_police", "name": "Dodge Charger Police", "type": "Gas"},
    {"id": "vehicle.jeep.wrangler_rubicon", "name": "Jeep Wrangler Rubicon", "type": "Gas"},
    {"id": "vehicle.harley-davidson.low_rider", "name": "Harley-Davidson Low Rider", "type": "Gas"},
    {"id": "vehicle.micro.microlino", "name": "Micro Microlino", "type": "Electric"},
    {"id": "vehicle.carlamotors.carlacola", "name": "Carlamotors Carlacola", "type": "Gas"},
    {"id": "vehicle.nissan.patrol", "name": "Nissan Patrol", "type": "Gas"},
    {"id": "vehicle.ford.crown", "name": "Ford Crown", "type": "Gas"},
    {"id": "vehicle.kawasaki.ninja", "name": "Kawasaki Ninja", "type": "Gas"},
    {"id": "vehicle.tesla.cybertruck", "name": "Tesla Cybertruck", "type": "Electric"},
    {"id": "vehicle.volkswagen.t2", "name": "Volkswagen T2", "type": "Gas"},
]

def main(page: ft.Page):
    page.title = "Vehicle App Launcher"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 800
    page.window_height = 600
    page.window_resizable = True

    # Initialize logs storage
    execution_logs = []
    
    # Vehicle dropdown
    vehicle_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option(f"{v['id']} - {v['name']}") for v in VEHICLES],
        label="Select Vehicle",
        width=300,
        autofocus=True,
    )

    # Status controls
    status_text = ft.Text("Ready to launch", size=16, color=ft.Colors.GREY_600)
    progress_ring = ft.ProgressRing(width=20, height=20, visible=False)

    # Logs table
    logs_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Time")),
            ft.DataColumn(ft.Text("Vehicle")),
            ft.DataColumn(ft.Text("Status")),
            ft.DataColumn(ft.Text("Command")),
        ],
        rows=[],
    )

    # Logs container with scroll
    logs_container = ft.Column(
        [ft.Text("Execution Logs", size=18, weight=ft.FontWeight.BOLD), logs_table],
        scroll=ft.ScrollMode.ALWAYS,
        height=300,
    )

    def add_log_entry(vehicle, status, command):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        execution_logs.insert(0, {
            "time": timestamp,
            "vehicle": vehicle,
            "status": status,
            "command": command
        })
        
        # Update table (show only last 20 entries)
        logs_table.rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(log["time"])),
                    ft.DataCell(ft.Text(log["vehicle"])),
                    ft.DataCell(ft.Text(log["status"], color=
                        ft.Colors.GREEN if log["status"] == "Success" else ft.Colors.RED
                    )),
                    ft.DataCell(ft.Text(log["command"])),
                ]
            ) for log in execution_logs[:20]
        ]
        page.update()

    def run_in_separate_terminal():
        if not vehicle_dropdown.value:
            status_text.value = "Please select a vehicle first!"
            status_text.color = ft.Colors.RED
            page.update()
            return

        # Get selected vehicle details
        vehicle_id = vehicle_dropdown.value.split(" - ")[0]
        vehicle_name = next(v["name"] for v in VEHICLES if v["id"] == vehicle_id)
        
        # Path to the external Python script
        script_path = os.path.join(os.path.dirname(__file__), "manual_control_carsim.py")
        
        if not os.path.exists(script_path):
            status_text.value = "Error: external_app.py not found!"
            status_text.color = ft.Colors.RED
            progress_ring.visible = False
            add_log_entry(vehicle_name, "Failed", "Script not found")
            page.update()
            return

        try:
            status_text.value = f"Launching for {vehicle_name}..."
            status_text.color = ft.Colors.BLUE
            progress_ring.visible = True
            page.update()

            # Prepare command with vehicle argument
            command = f'python "{script_path}"'
            
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
                    status_text.value = f"Launched {vehicle_name} successfully!"
                    status_text.color = ft.Colors.GREEN
                    add_log_entry(vehicle_name, "Success", command)
                except Exception as e:
                    status_text.value = f"Error launching: {str(e)}"
                    status_text.color = ft.Colors.RED
                    add_log_entry(vehicle_name, "Failed", str(e))
                finally:
                    progress_ring.visible = False
                    page.update()

            Thread(target=execute_command, daemon=True).start()
            
        except Exception as e:
            status_text.value = f"Error: {str(e)}"
            status_text.color = ft.Colors.RED
            progress_ring.visible = False
            add_log_entry(vehicle_name, "Failed", str(e))
            page.update()

    # Create the launch button
    launch_button = ft.ElevatedButton(
        content=ft.Row(
            [
                ft.Icon(ft.Icons.PLAY_ARROW),
                ft.Text("Launch Vehicle App", size=16),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=10,
        ),
        on_click=lambda e: run_in_separate_terminal(),
        width=220,
        height=50,
    )

    # Add controls to the page
    page.add(
        ft.Column(
            [
                ft.Row(
                    [
                        ft.Image(src="https://flet.dev/img/logo.svg", width=100),
                        ft.Text("Vehicle App Launcher", size=24, weight=ft.FontWeight.BOLD),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Divider(height=20),
                ft.Row(
                    [
                        vehicle_dropdown,
                        launch_button,
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=20,
                ),
                ft.Row(
                    [
                        progress_ring,
                        status_text,
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                ),
                ft.Divider(height=20),
                logs_container,
            ],
            spacing=10,
            expand=True,
        )
    )

if __name__ == "__main__":
    ft.app(target=main)