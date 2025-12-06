# module/realtime_data_collection.py
"""
Real-time data collection for speech analysis.
Collects:
1. Eye tracker data (focus/not focus)
2. Heart rate (BPM) from Arduino ESP32 via Serial
3. Calculates average HR over collection period
4. Calculates not_focus_count / total_time
"""

import os
import sys
import time
import threading
import serial
import serial.tools.list_ports
import re
from collections import deque
from typing import Optional, Dict, List, Tuple
import cv2

# Import gaze tracker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gaze_tracker import GazeTracker


class SerialHeartRateReader:
    """Read heart rate from Arduino ESP32 via Serial/USB."""
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.bpm_buffer = deque(maxlen=1000)  # Store up to 1000 readings
        self.running = False
        self._thread = None
        self.lock = threading.Lock()
        self.last_bpm = None
        
    def find_arduino_port(self) -> Optional[str]:
        """Auto-detect Arduino/ESP32 port."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc_lower = p.description.lower()
            if any(keyword in desc_lower for keyword in 
                   ['arduino', 'ch340', 'cp210', 'usb serial', 'esp32', 'silicon labs']):
                return p.device
        return None
    
    def start(self) -> bool:
        """Start reading from serial port."""
        if self.running:
            return True
        
        if self.port is None:
            self.port = self.find_arduino_port()
            if self.port is None:
                print("⚠️  Arduino port not found. Please specify port manually.")
                return False
        
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            self.running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"✅ Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Arduino: {e}")
            return False
    
    def _read_loop(self):
        """Background thread to continuously read serial data."""
        while self.running:
            try:
                if self.serial and self.serial.in_waiting > 0:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Parse BPM from lines like "💓 BPM: 85" or "BPM:85" or "🖐️ Wrist BPM: 85"
                    match = re.search(r'(\d+)', line)
                    if match:
                        bpm = int(match.group(1))
                        if 50 <= bpm <= 200:  # Sanity check
                            with self.lock:
                                self.bpm_buffer.append({
                                    'bpm': bpm,
                                    'timestamp': time.time()
                                })
                                self.last_bpm = bpm
            except Exception as e:
                if self.running:
                    print(f"⚠️  Serial read error: {e}")
            time.sleep(0.01)
    
    def get_latest_bpm(self) -> Optional[float]:
        """Get most recent BPM reading."""
        with self.lock:
            return self.last_bpm
    
    def get_bpm_in_range(self, start_time: float, end_time: float) -> List[Dict]:
        """Get all BPM readings within a time range."""
        with self.lock:
            return [
                reading for reading in self.bpm_buffer
                if start_time <= reading['timestamp'] <= end_time
            ]
    
    def get_avg_bpm_in_range(self, start_time: float, end_time: float) -> Optional[float]:
        """Get average BPM within a time range."""
        readings = self.get_bpm_in_range(start_time, end_time)
        if readings:
            bpm_values = [r['bpm'] for r in readings]
            return sum(bpm_values) / len(bpm_values)
        return None
    
    def stop(self):
        """Stop reading and close serial connection."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.serial:
            self.serial.close()


class RealTimeDataCollector:
    """Collects real-time heart rate and eye tracking data."""
    
    def __init__(self, 
                 camera_index: int = 0,
                 show_camera: bool = True,
                 serial_port: Optional[str] = None,
                 serial_baudrate: int = 115200):
        self.camera_index = camera_index
        self.show_camera = show_camera
        self.gaze_tracker = None
        self.hr_reader = SerialHeartRateReader(port=serial_port, baudrate=serial_baudrate)
        
        # Data storage
        self.eye_tracker_data = deque(maxlen=10000)  # Store focus states with timestamps
        self.collection_start_time = None
        self.collection_end_time = None
        self._stop_collection = False
        self._collection_lock = threading.Lock()
        self._previous_gaze_direction = None  # Track previous direction for movement calculation
        
    def start_collection(self):
        """Start collecting real-time data."""
        print("\n" + "="*60)
        print("🚀 Starting Real-Time Data Collection")
        print("="*60)
        
        self.collection_start_time = time.time()
        self._stop_collection = False
        
        # Start eye tracker
        print("\n👁️  Initializing eye tracker...")
        try:
            self.gaze_tracker = GazeTracker(
                camera_index=self.camera_index,
                show_window=self.show_camera,
                window_name="Eye Tracker - Real-Time Collection"
            )
            self.gaze_tracker.start(callback=self._on_gaze_update, display_info_callback=self._get_display_info)
            print("✅ Eye tracker started")
        except Exception as e:
            print(f"❌ Failed to start eye tracker: {e}")
            self.gaze_tracker = None
        
        # Start heart rate reader
        print("\n❤️  Initializing heart rate reader...")
        if not self.hr_reader.start():
            print("⚠️  Heart rate reader not available. Continuing without HR data.")
        
        print("\n" + "="*60)
        print("✅ Data collection started!")
        print("   - Press 'q' in camera window to stop")
        print("   - Or call stop_collection() method")
        print("="*60 + "\n")
        
    def _get_display_info(self) -> Dict:
        """Get current display information for camera overlay."""
        try:
            # Get current gaze direction
            current_direction = "unknown"
            if self.gaze_tracker and hasattr(self.gaze_tracker, '_gaze'):
                try:
                    gaze = self.gaze_tracker._gaze
                    if gaze.is_blinking():
                        current_direction = "blink"
                    elif gaze.is_right():
                        current_direction = "right"
                    elif gaze.is_left():
                        current_direction = "left"
                    elif gaze.is_center():
                        current_direction = "center"
                except:
                    pass
            
            # Get current statistics
            et_stats = self.get_eye_tracker_stats()
            hr_stats = self.get_heart_rate_stats()
            elapsed = self.get_collection_duration()
            
            return {
                'gaze_direction': current_direction,
                'movement_count': et_stats.get('movement_count', 0),
                'elapsed_time': elapsed,
                'heart_rate': hr_stats.get('latest_bpm'),
                'not_focus_periods': et_stats.get('not_focus_count', 0)
            }
        except:
            return {}
    
    def _on_gaze_update(self, focused: bool):
        """Callback when gaze state changes."""
        timestamp = time.time()
        
        # Detect gaze direction and blink state
        gaze_direction = "unknown"
        is_blinking = False
        movement_detected = False
        
        if self.gaze_tracker and hasattr(self.gaze_tracker, '_gaze'):
            try:
                gaze = self.gaze_tracker._gaze
                
                # Check blink state
                blink_state = gaze.is_blinking()
                is_blinking = blink_state is True
                
                # Detect gaze direction (only if not blinking)
                if not is_blinking:
                    if gaze.is_right():
                        gaze_direction = "right"
                    elif gaze.is_left():
                        gaze_direction = "left"
                    elif gaze.is_center():
                        gaze_direction = "center"
                    else:
                        gaze_direction = "unknown"
                else:
                    gaze_direction = "blink"
                
                # Calculate movement (direction change)
                if self._previous_gaze_direction is not None:
                    # Movement detected if direction changed (excluding blinks)
                    if (gaze_direction != self._previous_gaze_direction and 
                        gaze_direction != "blink" and 
                        self._previous_gaze_direction != "blink"):
                        movement_detected = True
                self._previous_gaze_direction = gaze_direction
                    
            except Exception as e:
                pass
        
        with self._collection_lock:
            self.eye_tracker_data.append({
                'timestamp': timestamp,
                'focused': focused,
                'blinking': is_blinking,
                'gaze_direction': gaze_direction,
                'movement': movement_detected
            })
    
    def stop_collection(self):
        """Stop collecting data."""
        print("\n🛑 Stopping data collection...")
        self._stop_collection = True
        self.collection_end_time = time.time()
        
        if self.gaze_tracker:
            self.gaze_tracker.stop()
            print("✅ Eye tracker stopped")
        
        self.hr_reader.stop()
        print("✅ Heart rate reader stopped")
        
        print("✅ Data collection stopped\n")
    
    def get_collection_duration(self) -> float:
        """Get total collection duration in seconds."""
        if self.collection_start_time is None:
            return 0.0
        end_time = self.collection_end_time if self.collection_end_time else time.time()
        return end_time - self.collection_start_time
    
    def get_eye_tracker_stats(self) -> Dict:
        """Get eye tracker statistics."""
        if not self.eye_tracker_data or self.collection_start_time is None:
            return {
                'not_focus_count': 0,
                'total_samples': 0,
                'focus_percentage': 0.0,
                'not_focus_percentage': 0.0,
                'not_focus_per_minute': 0.0,
                'blink_count': 0,
                'gaze_left_count': 0,
                'gaze_right_count': 0,
                'gaze_center_count': 0,
                'gaze_unknown_count': 0,
                'movement_count': 0
            }
        
        start_time = self.collection_start_time
        end_time = self.collection_end_time if self.collection_end_time else time.time()
        duration = end_time - start_time
        
        # Filter data within collection period
        with self._collection_lock:
            data_in_range = [
                d for d in self.eye_tracker_data
                if start_time <= d['timestamp'] <= end_time
            ]
        
        if not data_in_range:
            return {
                'not_focus_count': 0,
                'total_samples': 0,
                'focus_percentage': 0.0,
                'not_focus_percentage': 0.0,
                'not_focus_per_minute': 0.0,
                'blink_count': 0,
                'gaze_left_count': 0,
                'gaze_right_count': 0,
                'gaze_center_count': 0,
                'gaze_unknown_count': 0,
                'movement_count': 0
            }
        
        # Count blinks separately (blinks are not counted as "not focused")
        blink_count = sum(1 for d in data_in_range if d.get('blinking', False))
        
        # Count gaze directions
        gaze_left_count = sum(1 for d in data_in_range if d.get('gaze_direction') == 'left')
        gaze_right_count = sum(1 for d in data_in_range if d.get('gaze_direction') == 'right')
        gaze_center_count = sum(1 for d in data_in_range if d.get('gaze_direction') == 'center')
        gaze_unknown_count = sum(1 for d in data_in_range if d.get('gaze_direction') == 'unknown')
        
        # Count movements (direction changes)
        movement_count = sum(1 for d in data_in_range if d.get('movement', False))
        
        # Count "not focused" periods - only count if duration >= 5 seconds
        # Filter out blinks and identify consecutive "not focused" periods
        NOT_FOCUS_THRESHOLD_SECONDS = 2.0
        
        # Identify consecutive "not focused" periods (excluding blinks)
        not_focus_periods = []
        current_period_start = None
        
        for i, d in enumerate(data_in_range):
            is_not_focused = not d['focused'] and not d.get('blinking', False)
            
            if is_not_focused:
                if current_period_start is None:
                    # Start of a new "not focused" period
                    current_period_start = d['timestamp']
            else:
                if current_period_start is not None:
                    # End of a "not focused" period
                    period_duration = d['timestamp'] - current_period_start
                    if period_duration >= NOT_FOCUS_THRESHOLD_SECONDS:
                        not_focus_periods.append({
                            'start': current_period_start,
                            'end': d['timestamp'],
                            'duration': period_duration
                        })
                    current_period_start = None
        
        # Handle case where period continues until end of collection
        if current_period_start is not None:
            period_duration = end_time - current_period_start
            if period_duration >= NOT_FOCUS_THRESHOLD_SECONDS:
                not_focus_periods.append({
                    'start': current_period_start,
                    'end': end_time,
                    'duration': period_duration
                })
        
        # Count "not focused" periods (only those >= 5 seconds)
        not_focus_count = len(not_focus_periods)
        
        # Calculate total time spent in "not focus" periods (>= 5 seconds)
        total_not_focus_time = sum(p['duration'] for p in not_focus_periods)
        
        total_samples = len(data_in_range)
        focus_count = sum(1 for d in data_in_range if d['focused'] and not d.get('blinking', False))
        
        # Calculate percentages based on time, not sample counts
        focus_percentage = ((duration - total_not_focus_time) / duration * 100) if duration > 0 else 0.0
        not_focus_percentage = (total_not_focus_time / duration * 100) if duration > 0 else 0.0
        not_focus_per_minute = (not_focus_count / duration * 60) if duration > 0 else 0.0
        movement_per_minute = (movement_count / duration * 60) if duration > 0 else 0.0
        
        return {
            'not_focus_count': not_focus_count,
            'focus_count': focus_count,
            'blink_count': blink_count,
            'gaze_left_count': gaze_left_count,
            'gaze_right_count': gaze_right_count,
            'gaze_center_count': gaze_center_count,
            'gaze_unknown_count': gaze_unknown_count,
            'movement_count': movement_count,
            'movement_per_minute': movement_per_minute,
            'total_samples': total_samples,
            'focus_percentage': focus_percentage,
            'not_focus_percentage': not_focus_percentage,
            'not_focus_per_minute': not_focus_per_minute,
            'total_not_focus_time_seconds': total_not_focus_time,
            'duration_seconds': duration,
            'duration_minutes': duration / 60.0
        }
    
    def get_heart_rate_stats(self) -> Dict:
        """Get heart rate statistics."""
        if self.collection_start_time is None:
            return {
                'avg_bpm': None,
                'min_bpm': None,
                'max_bpm': None,
                'total_readings': 0
            }
        
        start_time = self.collection_start_time
        end_time = self.collection_end_time if self.collection_end_time else time.time()
        
        readings = self.hr_reader.get_bpm_in_range(start_time, end_time)
        
        if not readings:
            return {
                'avg_bpm': None,
                'min_bpm': None,
                'max_bpm': None,
                'total_readings': 0
            }
        
        bpm_values = [r['bpm'] for r in readings]
        
        return {
            'avg_bpm': sum(bpm_values) / len(bpm_values),
            'min_bpm': min(bpm_values),
            'max_bpm': max(bpm_values),
            'total_readings': len(bpm_values),
            'latest_bpm': self.hr_reader.get_latest_bpm()
        }
    
    def get_all_stats(self) -> Dict:
        """Get all collected statistics."""
        eye_stats = self.get_eye_tracker_stats()
        hr_stats = self.get_heart_rate_stats()
        
        return {
            'collection_duration_seconds': self.get_collection_duration(),
            'collection_duration_minutes': self.get_collection_duration() / 60.0,
            'collection_start_time': self.collection_start_time,
            'eye_tracker': eye_stats,
            'heart_rate': hr_stats
        }
    
    def print_summary(self):
        """Print summary of collected data."""
        stats = self.get_all_stats()
        
        print("\n" + "="*60)
        print("📊 DATA COLLECTION SUMMARY")
        print("="*60)
        
        print(f"\n⏱️  Collection Duration:")
        print(f"   • Total time: {stats['collection_duration_seconds']:.2f} seconds ({stats['collection_duration_minutes']:.2f} minutes)")
        
        print(f"\n👁️  Eye Tracker Statistics:")
        et = stats['eye_tracker']
        print(f"   • Total samples: {et['total_samples']}")
        print(f"   • Focus count: {et['focus_count']} ({et['focus_percentage']:.1f}% of time)")
        print(f"   • Not focus periods (>=5s): {et['not_focus_count']} ({et['not_focus_percentage']:.1f}% of time)")
        print(f"   • Total not focus time: {et.get('total_not_focus_time_seconds', 0):.1f} seconds")
        print(f"   • Blink count: {et.get('blink_count', 0)} (excluded from not focus)")
        print(f"   • Not focus periods per minute: {et['not_focus_per_minute']:.2f}")
        print(f"\n   📍 Gaze Direction:")
        print(f"      • Looking Left: {et.get('gaze_left_count', 0)}")
        print(f"      • Looking Right: {et.get('gaze_right_count', 0)}")
        print(f"      • Looking Center: {et.get('gaze_center_count', 0)}")
        print(f"      • Unknown: {et.get('gaze_unknown_count', 0)}")
        print(f"\n   🔄 Gaze Movement:")
        print(f"      • Total movements: {et.get('movement_count', 0)}")
        print(f"      • Movements per minute: {et.get('movement_per_minute', 0):.2f}")
        
        print(f"\n❤️  Heart Rate Statistics:")
        hr = stats['heart_rate']
        if hr['avg_bpm'] is not None:
            print(f"   • Average BPM: {hr['avg_bpm']:.1f}")
            print(f"   • Min BPM: {hr['min_bpm']:.0f}")
            print(f"   • Max BPM: {hr['max_bpm']:.0f}")
            print(f"   • Total readings: {hr['total_readings']}")
            if hr['latest_bpm']:
                print(f"   • Latest BPM: {hr['latest_bpm']:.0f}")
        else:
            print("   • No heart rate data available")
        
        print("="*60 + "\n")
        
        return stats


def collect_data_for_duration(duration_minutes: float = 3.0,
                              camera_index: int = 0,
                              show_camera: bool = True,
                              serial_port: Optional[str] = None) -> Dict:
    """
    Collect data for a specified duration.
    
    Args:
        duration_minutes: Duration to collect data (default: 3 minutes)
        camera_index: Camera index (default: 0)
        show_camera: Whether to show camera window (default: True)
        serial_port: Serial port for Arduino (None = auto-detect)
    
    Returns:
        Dictionary with collected statistics
    """
    collector = RealTimeDataCollector(
        camera_index=camera_index,
        show_camera=show_camera,
        serial_port=serial_port
    )
    
    try:
        # Start collection
        collector.start_collection()
        
        # Wait for specified duration or until user stops
        duration_seconds = duration_minutes * 60
        start_time = time.time()
        
        print(f"⏱️  Collecting data for {duration_minutes} minutes...")
        print("   (Press 'q' in camera window to stop early)\n")
        
        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time
            remaining = duration_seconds - elapsed
            
            # Print progress every 10 seconds
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                print(f"   ⏳ Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min", end='\r')
            
            # Check if user pressed 'q' in camera window
            if collector.gaze_tracker and not collector.gaze_tracker.is_running():
                print("\n   ⚠️  Collection stopped by user (camera window closed)")
                break
            
            time.sleep(1)
        
        print("\n   ✅ Collection duration reached!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user (Ctrl+C)")
    finally:
        # Stop collection
        collector.stop_collection()
        
        # Print summary
        stats = collector.print_summary()
        
        return stats


def collect_data_until_stopped(camera_index: int = 0,
                               show_camera: bool = True,
                               serial_port: Optional[str] = None) -> Dict:
    """
    Collect data until user stops (press 'q' in camera window or Ctrl+C).
    
    Args:
        camera_index: Camera index (default: 0)
        show_camera: Whether to show camera window (default: True)
        serial_port: Serial port for Arduino (None = auto-detect)
    
    Returns:
        Dictionary with collected statistics
    """
    collector = RealTimeDataCollector(
        camera_index=camera_index,
        show_camera=show_camera,
        serial_port=serial_port
    )
    
    try:
        # Start collection
        collector.start_collection()
        
        print("⏱️  Collecting data until stopped...")
        print("   (Press 'q' in camera window or Ctrl+C to stop)\n")
        
        # Keep running until stopped
        while True:
            # Check if user pressed 'q' in camera window
            if collector.gaze_tracker and not collector.gaze_tracker.is_running():
                print("\n   ⚠️  Collection stopped by user (camera window closed)")
                break
            
            # Print live stats every 5 seconds
            time.sleep(5)
            elapsed = collector.get_collection_duration()
            hr_latest = collector.hr_reader.get_latest_bpm()
            et_stats = collector.get_eye_tracker_stats()
            
            hr_display = f"{hr_latest:.0f}" if hr_latest else "N/A"
            # Get current gaze direction from latest data
            current_direction = "N/A"
            if collector.gaze_tracker and hasattr(collector.gaze_tracker, '_gaze'):
                try:
                    gaze = collector.gaze_tracker._gaze
                    if gaze.is_blinking():
                        current_direction = "Blink"
                    elif gaze.is_right():
                        current_direction = "Right"
                    elif gaze.is_left():
                        current_direction = "Left"
                    elif gaze.is_center():
                        current_direction = "Center"
                except:
                    pass
            
            print(f"   ⏳ Elapsed: {elapsed/60:.1f} min | "
                  f"HR: {hr_display} BPM | "
                  f"Gaze: {current_direction} | "
                  f"Movements: {et_stats.get('movement_count', 0)}", 
                  end='\r')
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user (Ctrl+C)")
    finally:
        # Stop collection
        collector.stop_collection()
        
        # Print summary
        stats = collector.print_summary()
        
        return stats


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time data collection for speech analysis')
    parser.add_argument('--duration', type=float, default=3.0,
                       help='Collection duration in minutes (default: 3.0)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (default: 0)')
    parser.add_argument('--no-camera-window', action='store_true',
                       help='Hide camera window')
    parser.add_argument('--port', type=str, default=None,
                       help='Serial port for Arduino (default: auto-detect)')
    parser.add_argument('--until-stopped', action='store_true',
                       help='Collect until manually stopped (ignore --duration)')
    
    args = parser.parse_args()
    
    if args.until_stopped:
        stats = collect_data_until_stopped(
            camera_index=args.camera,
            show_camera=not args.no_camera_window,
            serial_port=args.port
        )
    else:
        stats = collect_data_for_duration(
            duration_minutes=args.duration,
            camera_index=args.camera,
            show_camera=not args.no_camera_window,
            serial_port=args.port
        )
    
    # Return stats in format compatible with analysis code
    return stats


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("Real-Time Data Collection for Speech Analysis")
    print("="*60)
    print("\nChoose collection mode:")
    print("1. Collect for 3 minutes (default)")
    print("2. Collect until manually stopped")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    
    if choice == "2":
        stats = collect_data_until_stopped(serial_port="COM7")
    else:
        duration = input("Enter duration in minutes (default=3): ").strip()
        try:
            duration = float(duration) if duration else 3.0
        except ValueError:
            duration = 3.0
        stats = collect_data_for_duration(duration_minutes=duration, serial_port="COM7")
    
    # Print final results in format compatible with analysis code
    print("\n" + "="*60)
    print("📋 RESULTS (Compatible with analysis code)")
    print("="*60)
    
    # Heart rate results
    hr_avg = stats['heart_rate']['avg_bpm']
    if hr_avg is not None:
        print(f"avg_heart_rate_bpm: {hr_avg:.1f}")
    else:
        print("avg_heart_rate_bpm: None (no data collected)")
    
    # Eye tracker results
    et_stats = stats['eye_tracker']
    print(f"eye_tracker_not_focus_count: {et_stats['not_focus_count']}")
    print(f"collection_duration_minutes: {stats['collection_duration_minutes']:.2f}")
    print(f"not_focus_per_minute: {et_stats['not_focus_per_minute']:.2f}")
    print(f"\n📍 Gaze Direction Counts:")
    print(f"   gaze_left_count: {et_stats.get('gaze_left_count', 0)}")
    print(f"   gaze_right_count: {et_stats.get('gaze_right_count', 0)}")
    print(f"   gaze_center_count: {et_stats.get('gaze_center_count', 0)}")
    print(f"\n🔄 Gaze Movement:")
    print(f"   movement_count: {et_stats.get('movement_count', 0)}")
    print(f"   movement_per_minute: {et_stats.get('movement_per_minute', 0):.2f}")
    
    print("="*60)
    print("\n✅ Data collection complete!")
    print("   You can now use these values in your speech analysis code.")
    print("="*60 + "\n")