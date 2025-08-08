import flet as ft
import json
import os
from datetime import datetime
import threading
import time

class DetectionLogsTab:
    def __init__(self):
        self.detection_logs_data = []
        self.file_path = "detection/vehicle_detections.json"
        self.last_modified = 0
        self.refresh_timer = None
        
    def build(self):
        # Create the tab content
        self.logs_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Camera", width=100)),
                ft.DataColumn(ft.Text("Vehicle Class", width=200)),
                ft.DataColumn(ft.Text("Probability", width=100)),
                ft.DataColumn(ft.Text("Timestamp", width=150)),
                ft.DataColumn(ft.Text("Image", width=100)),
            ],
            rows=[],
            width=1000,
            column_spacing=10
        )
        
        self.logs_count = ft.Text("Total detections: 0", size=14)
        self.refresh_button = ft.ElevatedButton(
            text="Refresh",
            icon=ft.Icons.REFRESH,
            on_click=lambda e: self.refresh_logs()
        )
        
        self.auto_refresh_switch = ft.Switch(
            label="Auto-refresh",
            value=True,
            on_change=lambda e: self.toggle_auto_refresh(e.control.value)
        )
        
        # Create the tab
        return ft.Tab(
            text="Car Tracking Logs",
            content=ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Text("Vehicle Detection Logs", size=20, weight=ft.FontWeight.BOLD),
                        self.refresh_button,
                        self.auto_refresh_switch,
                        self.logs_count
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Divider(),
                    ft.Container(
                        self.logs_table,
                        border=ft.border.all(1, ft.Colors.GREY_400),
                        border_radius=5,
                        padding=10,
                        height=500,
                    )
                ], scroll=ft.ScrollMode.AUTO),
                padding=20
            )
        )
    
    def load_logs(self):
        """Load detection logs from JSON file"""
        try:
            if os.path.exists(self.file_path):
                current_modified = os.path.getmtime(self.file_path)
                if current_modified > self.last_modified:
                    with open(self.file_path, 'r') as f:
                        self.detection_logs_data = json.load(f)
                    self.last_modified = current_modified
                    return True
            return False
        except Exception as e:
            print(f"Error loading logs: {e}")
            return False
    
    def update_table(self):
        """Update the table with current logs"""
        self.logs_table.rows.clear()
        
        for detection in self.detection_logs_data:
            camera = detection.get('camera_label', 'Unknown')
            vehicle_class = detection.get('predicted_class', 'Unknown')
            probability = f"{detection.get('probability', 0):.2%}"
            timestamp = detection.get('timestamp', '')
            
            # Format timestamp
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    pass
            
            # Get first image
            images = detection.get('crop_image', [])
            image_text = "No image" if not images else f"{len(images)} images"
            
            self.logs_table.rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(camera)),
                        ft.DataCell(ft.Text(vehicle_class)),
                        ft.DataCell(ft.Text(probability)),
                        ft.DataCell(ft.Text(timestamp)),
                        ft.DataCell(ft.Text(image_text)),
                    ]
                )
            )
        
        self.logs_count.value = f"Total detections: {len(self.detection_logs_data)}"
    
    def refresh_logs(self):
        """Refresh the logs display"""
        if self.load_logs():
            self.update_table()
    
    def toggle_auto_refresh(self, enabled):
        """Toggle auto-refresh functionality"""
        if enabled:
            self.start_auto_refresh()
        else:
            self.stop_auto_refresh()
    
    def start_auto_refresh(self):
        """Start auto-refresh timer"""
        if self.refresh_timer is None:
            self.refresh_timer = threading.Thread(target=self._auto_refresh_worker, daemon=True)
            self.refresh_timer.start()
    
    def stop_auto_refresh(self):
        """Stop auto-refresh timer"""
        if self.refresh_timer:
            self.refresh_timer = None
    
    def _auto_refresh_worker(self):
        """Background worker for auto-refresh"""
        while True:
            if self.refresh_timer is None:
                break
            self.refresh_logs()
            time.sleep(2)  # Check every 2 seconds
    
    def initialize(self):
        """Initialize the tab with data"""
        self.refresh_logs()
        self.start_auto_refresh()
