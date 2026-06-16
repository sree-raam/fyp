#!/usr/bin/env python3
"""
OpenClaw v13 data-collection stable controller.

Keeps the v6 working movement principle, fixed 9-cylinder board map, advanced
dashboard/features, and fail-safe benchmark continuation for immediate thesis data collection.

Production-oriented single-file system for evaluating latency-aware LLM task
planning and autonomous navigation in a ROS 2 TurtleBot3 Gazebo warehouse.

The PC runs Gazebo and publishes /odom, /scan, /tf and consumes /cmd_vel.
The Jetson runs this controller, Ollama, Flask, mission execution, metrics,
and benchmark export.
"""

import base64
import csv
import io
import json
import logging
import math
import os
import re
import statistics
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from flask import Flask, Response, jsonify, render_template_string, request, send_file

try:
    import psutil
except Exception:
    psutil = None

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, LaserScan
except Exception:
    rclpy = None
    Twist = Odometry = Image = LaserScan = object
    MultiThreadedExecutor = ReentrantCallbackGroup = Node = object
    qos_profile_sensor_data = None


LOG = logging.getLogger("openclaw")
logging.basicConfig(
    level=os.environ.get("OPENCLAW_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


class Config:
    FLASK_HOST = os.environ.get("OPENCLAW_HOST", "0.0.0.0")
    FLASK_PORT = int(os.environ.get("OPENCLAW_PORT", "5000"))

    ODOM_TOPIC = os.environ.get("OPENCLAW_ODOM_TOPIC", "/odom")
    SCAN_TOPIC = os.environ.get("OPENCLAW_SCAN_TOPIC", "/scan")
    REQUIRE_SCAN_FOR_RUN = os.environ.get("OPENCLAW_REQUIRE_SCAN", "0").lower() in ("1", "true", "yes")
    CMD_VEL_TOPIC = os.environ.get("OPENCLAW_CMD_VEL_TOPIC", "/cmd_vel")
    CAMERA_TOPIC = os.environ.get("OPENCLAW_CAMERA_TOPIC", "/camera/image_raw")

    HOME_X = float(os.environ.get("OPENCLAW_HOME_X", "-2.0"))
    HOME_Y = float(os.environ.get("OPENCLAW_HOME_Y", "-0.5"))
    HOME_YAW = float(os.environ.get("OPENCLAW_HOME_YAW", "0.0"))

    SUPPORTED_MODELS = ["gemma2:2b", "qwen2.5:3b", "deepseek-r1:1.5b"]
    DEFAULT_MODEL = os.environ.get("OPENCLAW_MODEL", "gemma2:2b")
    OLLAMA_GENERATE_URL = os.environ.get("OPENCLAW_OLLAMA_GENERATE_URL", "http://localhost:11434/api/generate")
    OLLAMA_TAGS_URL = os.environ.get("OPENCLAW_OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")
    OLLAMA_TIMEOUT_S = float(os.environ.get("OPENCLAW_OLLAMA_TIMEOUT", "60"))

    EXPECTED_CYLINDERS = int(os.environ.get("OPENCLAW_EXPECTED_CYLINDERS", "9"))
    DISCOVERY_TIMEOUT_S = float(os.environ.get("OPENCLAW_DISCOVERY_TIMEOUT", "180"))
    MISSION_TIMEOUT_S = float(os.environ.get("OPENCLAW_MISSION_TIMEOUT", "900"))
    RUNS_PER_MODEL = int(os.environ.get("OPENCLAW_RUNS_PER_MODEL", "5"))

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openclaw_data")
    RUN_DIR = os.path.join(DATA_DIR, "runs")
    REPORT_DIR = os.path.join(DATA_DIR, "reports")
    CSV_PATH = os.path.join(DATA_DIR, "benchmark_runs.csv")
    JSON_PATH = os.path.join(DATA_DIR, "benchmark_runs.json")

    MAX_LINEAR_SPEED = float(os.environ.get("OPENCLAW_MAX_LINEAR", "0.22"))
    MAX_ANGULAR_SPEED = float(os.environ.get("OPENCLAW_MAX_ANGULAR", "1.20"))
    MAX_LINEAR_ACCEL = float(os.environ.get("OPENCLAW_LINEAR_ACCEL", "0.16"))
    MAX_ANGULAR_ACCEL = float(os.environ.get("OPENCLAW_ANGULAR_ACCEL", "0.9"))
    ARRIVAL_RADIUS = float(os.environ.get("OPENCLAW_ARRIVAL_RADIUS", "0.35"))
    HOME_RADIUS = float(os.environ.get("OPENCLAW_HOME_RADIUS", "0.50"))
    WAYPOINT_TIMEOUT_S = float(os.environ.get("OPENCLAW_WAYPOINT_TIMEOUT", "90"))
    CONTROL_HZ = float(os.environ.get("OPENCLAW_CONTROL_HZ", "12"))
    LOOKAHEAD_DISTANCE = float(os.environ.get("OPENCLAW_LOOKAHEAD", "0.55"))
    TARGET_STANDOFF = float(os.environ.get("OPENCLAW_TARGET_STANDOFF", "0.75"))

    OBSTACLE_STOP_M = float(os.environ.get("OPENCLAW_OBSTACLE_STOP", "0.22"))
    OBSTACLE_SLOW_M = float(os.environ.get("OPENCLAW_OBSTACLE_SLOW", "0.50"))
    OBSTACLE_CLEAR_M = float(os.environ.get("OPENCLAW_OBSTACLE_CLEAR", "0.42"))
    COLLISION_RANGE_M = float(os.environ.get("OPENCLAW_COLLISION_RANGE", "0.18"))

    CYL_CLUSTER_GAP_M = float(os.environ.get("OPENCLAW_CYL_CLUSTER_GAP", "0.18"))
    CYL_MIN_POINTS = int(os.environ.get("OPENCLAW_CYL_MIN_POINTS", "4"))
    CYL_MAX_POINTS = int(os.environ.get("OPENCLAW_CYL_MAX_POINTS", "90"))
    CYL_MIN_DIAMETER_M = float(os.environ.get("OPENCLAW_CYL_MIN_DIAMETER", "0.07"))
    CYL_MAX_DIAMETER_M = float(os.environ.get("OPENCLAW_CYL_MAX_DIAMETER", "0.45"))
    CYL_MAX_RANGE_M = float(os.environ.get("OPENCLAW_CYL_MAX_RANGE", "4.0"))
    CYL_MATCH_DISTANCE_M = float(os.environ.get("OPENCLAW_CYL_MATCH_DISTANCE", "0.45"))
    CYL_CONFIRM_HITS = int(os.environ.get("OPENCLAW_CYL_CONFIRM_HITS", "1"))
    ROTATE_ENTRY_RAD = math.radians(float(os.environ.get("OPENCLAW_ROTATE_ENTRY_DEG", "12")))
    DRIVE_HEADING_KP = float(os.environ.get("OPENCLAW_DRIVE_HEADING_KP", "0.80"))
    DRIVE_HEADING_KD = float(os.environ.get("OPENCLAW_DRIVE_HEADING_KD", "0.12"))
    DRIVE_LP_ALPHA = float(os.environ.get("OPENCLAW_DRIVE_LP_ALPHA", "0.35"))
    CREEP_SPEED = float(os.environ.get("OPENCLAW_CREEP_SPEED", "0.08"))

    MJPEG_QUALITY = int(os.environ.get("OPENCLAW_MJPEG_QUALITY", "70"))


for directory in (Config.DATA_DIR, Config.RUN_DIR, Config.REPORT_DIR):
    os.makedirs(directory, exist_ok=True)


class MissionPhase(str, Enum):
    IDLE = "idle"
    DISCOVERING = "discovering"
    PLANNING = "planning"
    RUNNING = "running"
    STOPPED = "stopped"
    RETURNING_HOME = "returning_home"
    RESETTING = "resetting"
    COMPLETE = "complete"
    FAILED = "failed"


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quat_to_yaw(q: Any) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(points: Iterable[Tuple[float, float]]) -> float:
    total = 0.0
    prev = None
    for point in points:
        if prev is not None:
            total += distance(prev, point)
        prev = point
    return total


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    stamp: float = 0.0

    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Cylinder:
    label: str
    x: float
    y: float
    radius: float
    hits: int
    discovered: bool
    first_seen: float
    last_seen: float


@dataclass
class LidarDetection:
    x: float
    y: float
    radius: float
    points: int
    range_m: float


@dataclass
class RunMetrics:
    run_id: str
    model: str
    run_number: int
    command: str
    started_at: str
    completed_at: str = ""
    llm_latency: float = 0.0
    navigation_time: float = 0.0
    mission_time: float = 0.0
    distance: float = 0.0
    collisions: int = 0
    replan_count: int = 0
    success: bool = False
    failure_reason: str = ""
    cpu_avg: float = 0.0
    ram_avg: float = 0.0
    power_avg: float = 0.0
    return_home_success: bool = False
    route: List[str] = field(default_factory=list)
    llm_reasoning: str = ""
    trajectory: List[Dict[str, float]] = field(default_factory=list)


class ThreadSafeState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.phase = MissionPhase.IDLE
        self.model = Config.DEFAULT_MODEL if Config.DEFAULT_MODEL in Config.SUPPORTED_MODELS else Config.SUPPORTED_MODELS[0]
        self.command = "Inspect all cylinders and return home."
        self.llm_reasoning = ""
        self.generated_route: List[str] = []
        self.current_target = ""
        self.planned_route_xy: List[Tuple[float, float]] = []
        self.actual_route: List[Tuple[float, float]] = []
        self.heatmap: Dict[str, int] = {}
        self.logs: List[Dict[str, str]] = []
        self.results: List[RunMetrics] = []
        self.stop_requested = False
        self.resume_requested = False
        self.benchmark_active = False
        self.home_required_before_run = True
        self.last_failure = ""
        self.run_counters = {m: 0 for m in Config.SUPPORTED_MODELS}
        self.last_status: Dict[str, Any] = {}

    def log(self, message: str, level: str = "info") -> None:
        entry = {"time": datetime.now().strftime("%H:%M:%S"), "level": level, "message": message}
        with self.lock:
            self.logs.append(entry)
            self.logs = self.logs[-500:]
        getattr(LOG, level if level in ("debug", "info", "warning", "error") else "info")(message)

    def reset_transient(self, keep_results: bool = True) -> None:
        with self.lock:
            self.command = "Inspect all cylinders and return home."
            self.llm_reasoning = ""
            self.generated_route = []
            self.current_target = ""
            self.planned_route_xy = []
            self.actual_route = []
            self.heatmap = {}
            self.stop_requested = False
            self.resume_requested = False
            self.last_failure = ""
            if not keep_results:
                self.results = []

    def append_pose(self, pose: Pose2D) -> None:
        with self.lock:
            self.actual_route.append((pose.x, pose.y))
            if len(self.actual_route) > 12000:
                self.actual_route = self.actual_route[-12000:]
            key = f"{round(pose.x / 0.15) * 0.15:.2f},{round(pose.y / 0.15) * 0.15:.2f}"
            self.heatmap[key] = self.heatmap.get(key, 0) + 1


STATE = ThreadSafeState()


class MetricsSampler:
    def __init__(self) -> None:
        self.samples: List[Tuple[float, float, float]] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.samples = []
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> Tuple[float, float, float]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if not self.samples:
            return (0.0, 0.0, 0.0)
        cpu = statistics.mean(s[0] for s in self.samples)
        ram = statistics.mean(s[1] for s in self.samples)
        power = statistics.mean(s[2] for s in self.samples)
        return (cpu, ram, power)

    def _loop(self) -> None:
        while self.running:
            self.samples.append(read_system_metrics())
            time.sleep(1.0)


def read_system_metrics() -> Tuple[float, float, float]:
    cpu = psutil.cpu_percent(interval=None) if psutil else 0.0
    ram = psutil.virtual_memory().percent if psutil else 0.0
    power = read_jetson_power_watts()
    return (float(cpu), float(ram), float(power))


def read_jetson_power_watts() -> float:
    candidates = [
        "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/1-0040/iio_device/in_power0_input",
    ]
    values = []
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                values.append(float(handle.read().strip()) / 1000.0)
        except Exception as exc:
            LOG.debug("Jetson power sensor unavailable at %s: %s", path, exc)
    return sum(values) if values else 0.0


class CylinderRegistry:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._items: List[Cylinder] = []

    def clear(self) -> None:
        with self.lock:
            self._items = []

    def update(self, detections: List[LidarDetection]) -> None:
        now = time.time()
        with self.lock:
            for det in detections:
                match = self._find_match(det.x, det.y)
                if match:
                    alpha = 0.25
                    match.x = (1.0 - alpha) * match.x + alpha * det.x
                    match.y = (1.0 - alpha) * match.y + alpha * det.y
                    match.radius = (1.0 - alpha) * match.radius + alpha * det.radius
                    match.hits += 1
                    match.last_seen = now
                    if match.hits >= Config.CYL_CONFIRM_HITS:
                        match.discovered = True
                elif len(self._items) < Config.EXPECTED_CYLINDERS:
                    label = f"CYL_{len(self._items) + 1}"
                    self._items.append(
                        Cylinder(label=label, x=det.x, y=det.y, radius=det.radius, hits=1, discovered=False, first_seen=now, last_seen=now)
                    )
            self._items.sort(key=lambda c: c.label)

    def _find_match(self, x: float, y: float) -> Optional[Cylinder]:
        best = None
        best_d = Config.CYL_MATCH_DISTANCE_M
        for cyl in self._items:
            d = distance((x, y), (cyl.x, cyl.y))
            if d < best_d:
                best = cyl
                best_d = d
        return best

    def discovered(self) -> List[Cylinder]:
        with self.lock:
            return [Cylinder(**asdict(c)) for c in self._items if c.discovered]

    def all(self) -> List[Cylinder]:
        with self.lock:
            return [Cylinder(**asdict(c)) for c in self._items]

    def by_label(self, label: str) -> Optional[Cylinder]:
        with self.lock:
            for cyl in self._items:
                if cyl.label == label:
                    return Cylinder(**asdict(cyl))
        return None

    def ready(self) -> bool:
        return len(self.discovered()) >= Config.EXPECTED_CYLINDERS


CYLINDERS = CylinderRegistry()

# Fixed Gazebo layout used for stable FYP data collection.
# This removes the broken endless "exploring / waiting for cylinders" gate.
# Edit here or override using OPENCLAW_LAYOUT_JSON if your Gazebo coordinates differ.
DEFAULT_KNOWN_CYLINDERS = [
    # label, x, y, radius. 9 nodes based on your hex/warehouse board layout.
    ("CYL_1",  1.65, -0.95, 0.12),
    ("CYL_2",  1.65,  0.95, 0.12),
    ("CYL_3",  0.55,  1.85, 0.12),
    ("CYL_4", -0.85,  1.85, 0.12),
    ("CYL_5", -1.85,  0.55, 0.12),
    ("CYL_6", -1.85, -0.85, 0.12),
    ("CYL_7", -0.45, -1.75, 0.12),
    ("CYL_8",  0.95, -1.75, 0.12),
    ("CYL_9",  0.00,  0.00, 0.12),
]

def _load_known_cylinders() -> List[Tuple[str, float, float, float]]:
    raw = os.environ.get("OPENCLAW_LAYOUT_JSON", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            out = []
            for item in data:
                out.append((str(item["label"]).upper(), float(item["x"]), float(item["y"]), float(item.get("radius", 0.12))))
            if out:
                return out
        except Exception as exc:
            LOG.warning("OPENCLAW_LAYOUT_JSON invalid; using default known layout: %s", exc)
    return list(DEFAULT_KNOWN_CYLINDERS)

KNOWN_CYLINDERS = _load_known_cylinders()

def install_known_cylinders() -> None:
    """Install fixed confirmed cylinders into the existing registry.

    The old v6 movement worked, but the discovery state caused later versions to
    spin forever. For data collection, the Gazebo board is fixed, so the fair
    LLM test is route planning over the same known nodes, not discovering them.
    """
    with CYLINDERS.lock:
        CYLINDERS._items = []
        for idx, (label, x, y, radius) in enumerate(KNOWN_CYLINDERS, 1):
            CYLINDERS._items.append(Cylinder(
                label=label, x=float(x), y=float(y), radius=float(radius),
                hits=999, discovered=True, first_seen=time.time(), last_seen=time.time()
            ))


class LidarCylinderDetector:
    def detect(self, scan: Any, pose: Pose2D) -> List[LidarDetection]:
        if not scan or not hasattr(scan, "ranges"):
            return []
        points: List[Tuple[float, float, float]] = []
        angle = float(scan.angle_min)
        for r in scan.ranges:
            if math.isfinite(r) and 0.05 < r <= Config.CYL_MAX_RANGE_M:
                points.append((r * math.cos(angle), r * math.sin(angle), r))
            angle += float(scan.angle_increment)
        if len(points) < Config.CYL_MIN_POINTS:
            return []

        clusters: List[List[Tuple[float, float, float]]] = []
        current = [points[0]]
        for previous, point in zip(points, points[1:]):
            if math.hypot(point[0] - previous[0], point[1] - previous[1]) > Config.CYL_CLUSTER_GAP_M:
                clusters.append(current)
                current = []
            current.append(point)
        clusters.append(current)

        detections = []
        cy = math.cos(pose.yaw)
        sy = math.sin(pose.yaw)
        for cluster in clusters:
            if not (Config.CYL_MIN_POINTS <= len(cluster) <= Config.CYL_MAX_POINTS):
                continue
            fit = self._fit_circle([(p[0], p[1]) for p in cluster])
            if not fit:
                continue
            rx, ry, radius = fit
            diameter = 2.0 * radius
            if not (Config.CYL_MIN_DIAMETER_M <= diameter <= Config.CYL_MAX_DIAMETER_M):
                continue
            wx = pose.x + cy * rx - sy * ry
            wy = pose.y + sy * rx + cy * ry
            detections.append(LidarDetection(x=wx, y=wy, radius=radius, points=len(cluster), range_m=math.hypot(rx, ry)))
        return detections

    @staticmethod
    def _fit_circle(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        if len(points) < 3:
            return None
        n = len(points)
        mx = sum(x for x, _ in points) / n
        my = sum(y for _, y in points) / n
        u = [x - mx for x, _ in points]
        v = [y - my for _, y in points]
        suu = sum(x * x for x in u)
        svv = sum(y * y for y in v)
        suv = sum(x * y for x, y in zip(u, v))
        suuu = sum(x ** 3 for x in u)
        svvv = sum(y ** 3 for y in v)
        suvv = sum(x * y * y for x, y in zip(u, v))
        svuu = sum(y * x * x for x, y in zip(u, v))
        det = suu * svv - suv * suv
        if abs(det) < 1e-9:
            return None
        uc = (0.5 * (suuu + suvv) * svv - 0.5 * (svvv + svuu) * suv) / det
        vc = (0.5 * (svvv + svuu) * suu - 0.5 * (suuu + suvv) * suv) / det
        radius = math.sqrt(max(0.0, uc * uc + vc * vc + (suu + svv) / n))
        return (uc + mx, vc + my, radius)


class CameraBuffer:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jpeg: Optional[bytes] = None
        self.last_frame_time = 0.0
        self.frame_count = 0

    def update_ros_image(self, msg: Any) -> None:
        if cv2 is None or np is None:
            return
        try:
            height = int(msg.height)
            width = int(msg.width)
            data = np.frombuffer(msg.data, dtype=np.uint8)
            encoding = str(msg.encoding).lower()
            if encoding in ("rgb8", "bgr8"):
                channels = 3
                frame = data.reshape((height, width, channels))
                if encoding == "rgb8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif encoding in ("mono8", "8uc1"):
                frame = data.reshape((height, width))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif encoding in ("rgba8", "bgra8"):
                frame = data.reshape((height, width, 4))
                code = cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR
                frame = cv2.cvtColor(frame, code)
            else:
                return
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), Config.MJPEG_QUALITY])
            if ok:
                with self.lock:
                    self.jpeg = bytes(encoded)
                    self.last_frame_time = time.time()
                    self.frame_count += 1
        except Exception as exc:
            LOG.debug("camera frame ignored: %s", exc)

    def get_jpeg(self) -> bytes:
        with self.lock:
            if self.jpeg:
                return self.jpeg
        return self._fallback_jpeg()

    def status(self) -> Dict[str, Any]:
        with self.lock:
            age = time.time() - self.last_frame_time if self.last_frame_time else None
            return {"available": self.jpeg is not None, "age": age, "frames": self.frame_count, "topic": Config.CAMERA_TOPIC}

    def _fallback_jpeg(self) -> bytes:
        if cv2 is not None and np is not None:
            img = np.zeros((240, 420, 3), dtype=np.uint8)
            img[:, :] = (28, 34, 44)
            cv2.putText(img, "Gazebo camera topic unavailable", (28, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 235, 242), 1)
            cv2.putText(img, Config.CAMERA_TOPIC, (28, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 190, 255), 1)
            ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), Config.MJPEG_QUALITY])
            if ok:
                return bytes(encoded)
        payload = base64.b64decode(
            b"/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAAQABADASIAAhEBAxEB/8QAFwAAAwEAAAAAAAAAAAAAAAAAAAIEBf/EAB0QAAICAgMBAAAAAAAAAAAAAAECAAMRBBIhMUH/xAAVAQEBAAAAAAAAAAAAAAAAAAAAAf/EABYRAQEBAAAAAAAAAAAAAAAAAAABEf/aAAwDAQACEQMRAD8A0umzS3UbFuQh5x7B3Ehm2yK0T2m9iUtN+oYk2MTMhQcn/2Q=="
        )
        return payload


CAMERA = CameraBuffer()


class OpenClawRosNode(Node):
    def __init__(self) -> None:
        super().__init__("openclaw_benchmark_controller")
        self.group = ReentrantCallbackGroup()
        self.cmd_pub = self.create_publisher(Twist, Config.CMD_VEL_TOPIC, 10)
        self.create_subscription(Odometry, Config.ODOM_TOPIC, self._on_odom, 20, callback_group=self.group)
        self.create_subscription(LaserScan, Config.SCAN_TOPIC, self._on_scan, qos_profile_sensor_data, callback_group=self.group)
        self.create_subscription(Image, Config.CAMERA_TOPIC, self._on_camera, qos_profile_sensor_data, callback_group=self.group)
        self.pose = Pose2D(stamp=time.time())
        self.last_scan: Optional[Any] = None
        self.last_odom_time = 0.0
        self.last_scan_time = 0.0
        self.detector = LidarCylinderDetector()
        self.collision_count = 0
        self._last_collision_time = 0.0
        self._last_cmd = (0.0, 0.0)
        self._lock = threading.RLock()

    def _on_odom(self, msg: Any) -> None:
        p = msg.pose.pose.position
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        with self._lock:
            self.pose = Pose2D(float(p.x), float(p.y), yaw, time.time())
            self.last_odom_time = time.time()
        STATE.append_pose(self.pose)

    def _on_scan(self, msg: Any) -> None:
        with self._lock:
            self.last_scan = msg
            self.last_scan_time = time.time()
            pose = Pose2D(**asdict(self.pose))
        detections = self.detector.detect(msg, pose)
        if detections:
            CYLINDERS.update(detections)
        self._update_collision_count(msg)

    def _on_camera(self, msg: Any) -> None:
        CAMERA.update_ros_image(msg)

    def _update_collision_count(self, scan: Any) -> None:
        front = self.front_clearance(scan, arc_deg=45)
        if front < Config.COLLISION_RANGE_M and time.time() - self._last_collision_time > 1.5:
            self.collision_count += 1
            self._last_collision_time = time.time()
            STATE.log(f"Collision proximity event counted at {front:.2f} m", "warning")

    def get_pose(self) -> Pose2D:
        with self._lock:
            return Pose2D(**asdict(self.pose))

    def get_scan(self) -> Optional[Any]:
        with self._lock:
            return self.last_scan

    def publish_cmd(self, linear: float, angular: float) -> None:
        if rclpy is None:
            return
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)
        self._last_cmd = (float(linear), float(angular))

    def stop_robot(self) -> None:
        for _ in range(5):
            self.publish_cmd(0.0, 0.0)
            time.sleep(0.04)

    def front_clearance(self, scan: Optional[Any] = None, arc_deg: float = 55) -> float:
        scan = scan or self.get_scan()
        if not scan or not hasattr(scan, "ranges"):
            return float("inf")
        half = math.radians(arc_deg) / 2.0
        angle = float(scan.angle_min)
        best = float("inf")
        for r in scan.ranges:
            if abs(angle) <= half and math.isfinite(r):
                best = min(best, float(r))
            angle += float(scan.angle_increment)
        return best

    def side_clearance(self, side: str, scan: Optional[Any] = None) -> float:
        scan = scan or self.get_scan()
        if not scan or not hasattr(scan, "ranges"):
            return float("inf")
        if side == "left":
            low, high = math.radians(35), math.radians(110)
        else:
            low, high = math.radians(-110), math.radians(-35)
        angle = float(scan.angle_min)
        best = float("inf")
        for r in scan.ranges:
            if low <= angle <= high and math.isfinite(r):
                best = min(best, float(r))
            angle += float(scan.angle_increment)
        return best

    def ros_status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "available": True,
            "odom_topic": Config.ODOM_TOPIC,
            "scan_topic": Config.SCAN_TOPIC,
            "cmd_vel_topic": Config.CMD_VEL_TOPIC,
            "odom_age": now - self.last_odom_time if self.last_odom_time else None,
            "scan_age": now - self.last_scan_time if self.last_scan_time else None,
            "pose": asdict(self.get_pose()),
            "collisions": self.collision_count,
        }


ROS_NODE: Optional[OpenClawRosNode] = None


class SmoothNavigator:
    """Stable two-phase differential-drive navigator.

    Phase 1 rotates the robot in place until it faces the target. Phase 2 drives
    forward with small PD heading correction. This prevents the old duck-walking
    behaviour caused by turning strongly while moving forward.
    """
    def __init__(self, node: OpenClawRosNode) -> None:
        self.node = node
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.prev_heading_error = 0.0
        self.filtered_angular = 0.0
        self.replan_count = 0

    def reset_buffers(self) -> None:
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.prev_heading_error = 0.0
        self.filtered_angular = 0.0
        self.replan_count = 0

    def drive_to(self, goal: Tuple[float, float], arrival_radius: float, timeout_s: float, target_label: str = "") -> bool:
        start = time.time()
        dt = 1.0 / Config.CONTROL_HZ
        stable_arrivals = 0
        mode = "rotate"
        self.prev_heading_error = 0.0
        self.filtered_angular = 0.0
        while time.time() - start < timeout_s:
            if STATE.stop_requested:
                self.node.stop_robot()
                return False
            pose = self.node.get_pose()
            dist = distance(pose.xy(), goal)
            if dist <= arrival_radius:
                stable_arrivals += 1
                if stable_arrivals >= 4:
                    self.node.stop_robot()
                    return True
            else:
                stable_arrivals = 0

            desired_heading = math.atan2(goal[1] - pose.y, goal[0] - pose.x)
            heading_error = angle_wrap(desired_heading - pose.yaw)
            front = self.node.front_clearance()
            left = self.node.side_clearance("left")
            right = self.node.side_clearance("right")

            if abs(heading_error) > Config.ROTATE_ENTRY_RAD and dist > arrival_radius * 1.15:
                mode = "rotate"
            elif abs(heading_error) < Config.ROTATE_ENTRY_RAD * 0.65:
                mode = "drive"

            if front < Config.OBSTACLE_STOP_M:
                linear = 0.0
                angular = self._avoidance_turn(left, right)
                self.replan_count += 1
                mode = "rotate"
            elif mode == "rotate":
                linear = 0.0
                angular = clamp(2.2 * heading_error, -Config.MAX_ANGULAR_SPEED, Config.MAX_ANGULAR_SPEED)
            else:
                # Drive phase: gentle PD heading correction only.
                derr = (heading_error - self.prev_heading_error) / max(dt, 1e-3)
                raw_ang = Config.DRIVE_HEADING_KP * heading_error + Config.DRIVE_HEADING_KD * derr
                raw_ang = clamp(raw_ang, -0.55, 0.55)
                self.filtered_angular = (1.0 - Config.DRIVE_LP_ALPHA) * self.filtered_angular + Config.DRIVE_LP_ALPHA * raw_ang
                angular = self.filtered_angular
                linear = self._linear_profile(dist, heading_error, front)
                if 0.0 < linear < Config.CREEP_SPEED:
                    linear = Config.CREEP_SPEED
                if front < Config.OBSTACLE_CLEAR_M:
                    linear *= 0.35
                    angular += 0.25 * self._avoidance_turn(left, right)
                elif front < Config.OBSTACLE_SLOW_M:
                    linear *= 0.65

            self.prev_heading_error = heading_error
            linear, angular = self._limit_accel(linear, angular, dt)
            self.node.publish_cmd(linear, angular)
            time.sleep(dt)
        self.node.stop_robot()
        STATE.log(f"Timed out driving to {target_label or goal}", "warning")
        return False

    def follow_route(self, waypoints: List[Tuple[float, float]], labels: Optional[List[str]] = None) -> bool:
        labels = labels or ["" for _ in waypoints]
        for waypoint, label in zip(waypoints, labels):
            with STATE.lock:
                STATE.current_target = label
            if not self.drive_to(waypoint, Config.ARRIVAL_RADIUS, Config.WAYPOINT_TIMEOUT_S, label):
                return False
        return True

    def perception_sweep(self, duration_s: float = 12.0) -> None:
        start = time.time()
        dt = 1.0 / Config.CONTROL_HZ
        while time.time() - start < duration_s:
            if STATE.stop_requested or CYLINDERS.ready():
                break
            _, angular = self._limit_accel(0.0, 0.42, dt)
            self.node.publish_cmd(0.0, angular)
            time.sleep(dt)
        self.node.stop_robot()

    def return_home(self) -> bool:
        with STATE.lock:
            STATE.phase = MissionPhase.RETURNING_HOME
            STATE.current_target = "HOME"
        ok = False
        for attempt in range(2):
            ok = self.drive_to((Config.HOME_X, Config.HOME_Y), Config.HOME_RADIUS, 60.0, f"HOME attempt {attempt+1}")
            pose = self.node.get_pose()
            verified = distance(pose.xy(), (Config.HOME_X, Config.HOME_Y)) <= Config.HOME_RADIUS
            if ok and verified:
                self.align_heading(Config.HOME_YAW, timeout_s=8.0)
                self.node.stop_robot()
                with STATE.lock:
                    STATE.home_required_before_run = False
                return True
            STATE.log(f"HOME verification failed after attempt {attempt+1}; retrying navigation fallback.", "warning")
        self.node.stop_robot()
        return False

    def align_heading(self, yaw: float, timeout_s: float) -> bool:
        start = time.time()
        dt = 1.0 / Config.CONTROL_HZ
        while time.time() - start < timeout_s:
            err = angle_wrap(yaw - self.node.get_pose().yaw)
            if abs(err) < 0.07:
                self.node.stop_robot()
                return True
            _, angular = self._limit_accel(0.0, clamp(1.3 * err, -0.65, 0.65), dt)
            self.node.publish_cmd(0.0, angular)
            time.sleep(dt)
        self.node.stop_robot()
        return False

    def inspection_pose_for(self, cyl: Cylinder) -> Tuple[float, float]:
        home_vec = (Config.HOME_X - cyl.x, Config.HOME_Y - cyl.y)
        norm = math.hypot(home_vec[0], home_vec[1])
        if norm < 0.01:
            return (cyl.x - Config.TARGET_STANDOFF, cyl.y)
        return (cyl.x + home_vec[0] / norm * Config.TARGET_STANDOFF, cyl.y + home_vec[1] / norm * Config.TARGET_STANDOFF)

    def _linear_profile(self, dist: float, heading_error: float, front: float) -> float:
        heading_factor = clamp(1.0 - abs(heading_error) / 0.75, 0.25, 1.0)
        target_factor = clamp((dist - Config.ARRIVAL_RADIUS) / 0.75, 0.0, 1.0)
        obstacle_factor = 1.0
        if front < Config.OBSTACLE_SLOW_M:
            obstacle_factor = clamp((front - Config.OBSTACLE_STOP_M) / max(0.01, Config.OBSTACLE_SLOW_M - Config.OBSTACLE_STOP_M), 0.0, 1.0)
        return Config.MAX_LINEAR_SPEED * heading_factor * target_factor * obstacle_factor

    def _avoidance_turn(self, left: float, right: float) -> float:
        return 0.70 if left >= right else -0.70

    def _limit_accel(self, linear: float, angular: float, dt: float) -> Tuple[float, float]:
        lin_step = Config.MAX_LINEAR_ACCEL * dt
        ang_step = Config.MAX_ANGULAR_ACCEL * dt
        linear = clamp(linear, self.last_linear - lin_step, self.last_linear + lin_step)
        angular = clamp(angular, self.last_angular - ang_step, self.last_angular + ang_step)
        self.last_linear = linear
        self.last_angular = angular
        return linear, angular


NAVIGATOR: Optional[SmoothNavigator] = None


class OllamaPlanner:
    def available_models(self) -> List[str]:
        try:
            response = requests.get(Config.OLLAMA_TAGS_URL, timeout=3)
            response.raise_for_status()
            data = response.json()
            names = [m.get("name", "") for m in data.get("models", [])]
            return [m for m in Config.SUPPORTED_MODELS if m in names]
        except Exception:
            return []

    def status(self) -> Dict[str, Any]:
        available = self.available_models()
        return {"online": bool(available), "available_models": available, "supported_models": Config.SUPPORTED_MODELS}

    def plan(self, model: str, command: str, cylinders: List[Cylinder], pose: Pose2D) -> Dict[str, Any]:
        if model not in Config.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        labels = [c.label for c in cylinders]
        prompt = self._prompt(command, cylinders, pose)
        started = time.perf_counter()
        text = ""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096},
            }
            response = requests.post(Config.OLLAMA_GENERATE_URL, json=payload, timeout=Config.OLLAMA_TIMEOUT_S)
            response.raise_for_status()
            text = response.json().get("response", "")
        except Exception as exc:
            latency = time.perf_counter() - started
            fallback = self._nearest_neighbor(labels, cylinders, pose.xy())
            return {
                "route": fallback,
                "reasoning": f"Ollama planning failed ({exc}). Deterministic nearest-neighbor route used to preserve benchmark run.",
                "latency": latency,
                "raw": text,
                "fallback": True,
            }
        latency = time.perf_counter() - started
        route, reasoning = self._parse(text, labels)
        wants_all = any(w in command.lower() for w in ("all", "every", "complete", "whole", "full"))
        if route and wants_all:
            missing = [label for label in labels if label not in route]
            if missing:
                route.extend(missing)
        if not route:
            route = self._nearest_neighbor(labels, cylinders, pose.xy())
            reasoning = "LLM response did not contain a valid route. Deterministic nearest-neighbor route used."
        return {"route": route, "reasoning": reasoning, "latency": latency, "raw": text, "fallback": False}

    def _prompt(self, command: str, cylinders: List[Cylinder], pose: Pose2D) -> str:
        cyl_lines = "\n".join(f"- {c.label}: x={c.x:.2f}, y={c.y:.2f}" for c in cylinders)
        return f"""
You are the task planner for a TurtleBot3 warehouse inspection benchmark.
The robot already discovered all cylinders with LiDAR. You must only choose
inspection order and route strategy. Do not output motor commands, PID logic,
or obstacle avoidance logic.

Supported task: inspect discovered cylinders and return home.
Home: x={Config.HOME_X:.2f}, y={Config.HOME_Y:.2f}
Current robot pose: x={pose.x:.2f}, y={pose.y:.2f}, yaw={pose.yaw:.2f}
Discovered cylinders:
{cyl_lines}

User command: {command}

Return strict JSON only:
{{
  "reasoning": "brief route strategy focused on latency and efficiency",
  "route": ["CYL_1", "CYL_2", "CYL_3", "CYL_4", "CYL_5", "CYL_6", "CYL_7", "CYL_8", "CYL_9"]
}}
If the command asks for all/every/complete inspection, include every listed cylinder exactly once. If the command asks for nearest, one, two, left, right, or a named cylinder, include only the required cylinder labels. Never include HOME in the route because the robot returns home automatically.
"""

    def _parse(self, text: str, labels: List[str]) -> Tuple[List[str], str]:
        reasoning = text.strip()
        route: List[str] = []
        try:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if match:
                obj = json.loads(match.group(0))
                reasoning = str(obj.get("reasoning", "")).strip() or reasoning
                raw_route = obj.get("route", [])
                if isinstance(raw_route, list):
                    route = [str(x).strip().upper() for x in raw_route]
        except Exception as exc:
            LOG.debug("LLM JSON parse failed; falling back to route token extraction: %s", exc)
        if not route:
            route = [x.upper() for x in re.findall(r"CYL_[1-9]", text.upper())]
        normalized = []
        for label in route:
            if label in labels and label not in normalized:
                normalized.append(label)
        # Return only what the LLM requested. For "inspect all" missions,
        # plan() appends missing labels fairly after parsing. This prevents
        # a command such as "inspect nearest two" from being forced into all 9.
        return normalized, reasoning

    def _nearest_neighbor(self, labels: List[str], cylinders: List[Cylinder], start: Tuple[float, float]) -> List[str]:
        remaining = {c.label: c for c in cylinders}
        route = []
        current = start
        while remaining:
            label, cyl = min(remaining.items(), key=lambda item: distance(current, (item[1].x, item[1].y)))
            route.append(label)
            current = (cyl.x, cyl.y)
            remaining.pop(label)
        return [label for label in route if label in labels]


PLANNER = OllamaPlanner()


class DataStore:
    CSV_FIELDS = [
        "run_id",
        "model",
        "run_number",
        "command",
        "started_at",
        "completed_at",
        "llm_latency",
        "navigation_time",
        "mission_time",
        "distance",
        "collisions",
        "replan_count",
        "success",
        "failure_reason",
        "cpu_avg",
        "ram_avg",
        "power_avg",
        "return_home_success",
        "route",
        "llm_reasoning",
    ]

    @classmethod
    def append(cls, result: RunMetrics) -> None:
        exists = cls._csv_header_matches()
        with open(Config.CSV_PATH, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.CSV_FIELDS)
            if not exists:
                writer.writeheader()
            row = {field: getattr(result, field) for field in cls.CSV_FIELDS}
            row["route"] = " -> ".join(result.route)
            writer.writerow(row)
        cls._save_run_json(result)
        cls._rewrite_all_json()

    @classmethod
    def _csv_header_matches(cls) -> bool:
        if not os.path.exists(Config.CSV_PATH):
            return False
        try:
            with open(Config.CSV_PATH, "r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
            return header == cls.CSV_FIELDS
        except Exception as exc:
            LOG.warning("CSV header check failed, writing a new header: %s", exc)
            return False

    @classmethod
    def _save_run_json(cls, result: RunMetrics) -> None:
        path = os.path.join(Config.RUN_DIR, f"{result.run_id}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(result), handle, indent=2)

    @classmethod
    def _rewrite_all_json(cls) -> None:
        results = [asdict(r) for r in STATE.results]
        with open(Config.JSON_PATH, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)

    @classmethod
    def load_existing(cls) -> List[RunMetrics]:
        loaded = []
        if not os.path.isdir(Config.RUN_DIR):
            return loaded
        for name in sorted(os.listdir(Config.RUN_DIR)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(Config.RUN_DIR, name), "r", encoding="utf-8") as handle:
                    loaded.append(RunMetrics(**json.load(handle)))
            except Exception as exc:
                LOG.warning("Skipping unreadable run file %s: %s", name, exc)
        return loaded


def aggregate_results(results: List[RunMetrics]) -> Dict[str, Any]:
    by_model: Dict[str, List[RunMetrics]] = {m: [] for m in Config.SUPPORTED_MODELS}
    for result in results:
        if result.model in by_model:
            by_model[result.model].append(result)
    return {model: summarize_model(items) for model, items in by_model.items()}


def summarize_model(items: List[RunMetrics]) -> Dict[str, Any]:
    def stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        return {
            "avg": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }

    return {
        "runs": len(items),
        "success_rate": (sum(1 for r in items if r.success) / len(items) * 100.0) if items else 0.0,
        "llm_latency": stats([r.llm_latency for r in items]),
        "navigation_time": stats([r.navigation_time for r in items]),
        "mission_time": stats([r.mission_time for r in items]),
        "distance": stats([r.distance for r in items]),
        "collisions": stats([float(r.collisions) for r in items]),
        "power": stats([r.power_avg for r in items]),
    }


class MissionEngine:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.worker: Optional[threading.Thread] = None
        self.model_switching = False

    def busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def start_single(self, model: str, command: str) -> bool:
        with self.lock:
            if self.busy() or self.model_switching:
                return False
            if model not in Config.SUPPORTED_MODELS:
                return False
            with STATE.lock:
                STATE.model = model
            self.worker = threading.Thread(target=self._run_single_safe, args=(model, command), daemon=True)
            self.worker.start()
            return True

    def start_benchmark(self, models: List[str], runs: int, command: str) -> bool:
        with self.lock:
            if self.busy() or self.model_switching:
                return False
            self.worker = threading.Thread(target=self._run_benchmark_safe, args=(models, runs, command), daemon=True)
            self.worker.start()
            return True

    def stop(self) -> None:
        with STATE.lock:
            STATE.stop_requested = True
            STATE.phase = MissionPhase.STOPPED
        if ROS_NODE:
            ROS_NODE.stop_robot()
        STATE.log("Stop requested. Mission cancelled and collected data preserved.", "warning")

    def resume(self) -> None:
        with STATE.lock:
            STATE.stop_requested = False
            STATE.resume_requested = True
        STATE.log("Resume requested.", "info")

    def return_home(self) -> bool:
        if not NAVIGATOR:
            return False
        # Abort any current mission first so HOME command is not fighting with an active route.
        if self.busy():
            with STATE.lock:
                STATE.stop_requested = True
            if ROS_NODE:
                ROS_NODE.stop_robot()
            t0 = time.time()
            while self.busy() and time.time() - t0 < 3.0:
                time.sleep(0.05)
        with STATE.lock:
            STATE.stop_requested = False
            STATE.phase = MissionPhase.RETURNING_HOME
        ok = NAVIGATOR.return_home()
        with STATE.lock:
            STATE.phase = MissionPhase.IDLE if ok else MissionPhase.FAILED
            STATE.last_failure = "" if ok else "HOME return failed"
        return ok

    def switch_model(self, model: str) -> Tuple[bool, str]:
        if model not in Config.SUPPORTED_MODELS:
            return False, "unsupported model"
        with self.lock:
            if self.busy():
                return False, "mission is running"
        self._reset_internal_state(clear_cylinders=False)
        with STATE.lock:
            STATE.model = model
            STATE.phase = MissionPhase.IDLE
            STATE.home_required_before_run = True
            STATE.last_failure = ""
        STATE.log(f"Model selected: {model}. Next run will return HOME before collecting data.", "info")
        return True, "model selected"

    def _wait_for_ros_inputs(self, timeout_s: float = 8.0) -> bool:
        """Wait for ROS input without blocking data collection forever.

        v10/v11/v12 could abort before moving if /scan was late/stale even
        though /odom and /cmd_vel were usable. For the FYP benchmark, odometry
        is mandatory for movement/metrics. LiDAR is still used for avoidance
        when available, but by default it is not allowed to stop the whole run.
        Set OPENCLAW_REQUIRE_SCAN=1 if you want the strict old behaviour.
        """
        if not ROS_NODE:
            return False
        start = time.time()
        last_ros = {}
        while time.time() - start < timeout_s:
            if STATE.stop_requested:
                return False
            ros = ROS_NODE.ros_status()
            last_ros = ros
            odom_ok = ros.get("odom_age") is not None and ros["odom_age"] < 2.5
            scan_ok = ros.get("scan_age") is not None and ros["scan_age"] < 2.5
            if odom_ok and (scan_ok or not Config.REQUIRE_SCAN_FOR_RUN):
                if not scan_ok:
                    STATE.log("/scan not fresh; continuing with odom + fixed board map. LiDAR avoidance resumes when scan returns.", "warning")
                return True
            time.sleep(0.1)
        STATE.log(f"ROS input wait timeout. Last status: {last_ros}", "warning")
        return False

    def _ensure_home_before_run(self) -> bool:
        if not NAVIGATOR or not ROS_NODE:
            return False
        if not self._wait_for_ros_inputs():
            STATE.log("Cannot start: /odom is not active. Check ROS_DOMAIN_ID and Gazebo.", "warning")
            return False
        pose = ROS_NODE.get_pose()
        if distance(pose.xy(), (Config.HOME_X, Config.HOME_Y)) <= Config.HOME_RADIUS:
            ROS_NODE.stop_robot()
            with STATE.lock:
                STATE.home_required_before_run = False
            return True
        STATE.log("Returning HOME before starting the run.", "info")
        ok = NAVIGATOR.return_home()
        with STATE.lock:
            STATE.home_required_before_run = not ok
        if not ok:
            STATE.log("HOME verification failed. Continuing run from current pose and logging home failure instead of aborting data collection.", "warning")
        return True

    def reset_after_completion(self) -> bool:
        if not NAVIGATOR:
            return False
        with STATE.lock:
            STATE.phase = MissionPhase.RESETTING
            STATE.stop_requested = False
        home_ok = NAVIGATOR.return_home()
        if home_ok:
            self._reset_internal_state(clear_cylinders=False)
            with STATE.lock:
                STATE.phase = MissionPhase.IDLE
            STATE.log("Reset complete. Robot is at HOME and ready for the next run.", "info")
        else:
            with STATE.lock:
                STATE.phase = MissionPhase.FAILED
                STATE.last_failure = "reset return-home failed"
            STATE.log("Reset failed because HOME could not be verified.", "error")
        return home_ok

    def _run_single_safe(self, model: str, command: str) -> None:
        try:
            self.run_single(model, command)
        except Exception as exc:
            STATE.log(f"Mission failed: {exc}", "error")
            with STATE.lock:
                STATE.phase = MissionPhase.FAILED
                STATE.last_failure = str(exc)
            if ROS_NODE:
                ROS_NODE.stop_robot()

    def _run_benchmark_safe(self, models: List[str], runs: int, command: str) -> None:
        with STATE.lock:
            STATE.benchmark_active = True
        try:
            for model in models:
                if model not in Config.SUPPORTED_MODELS:
                    continue
                ok, msg = self.switch_model(model)
                if not ok:
                    STATE.log(f"Benchmark model switch failed for {model}: {msg}", "error")
                    break
                for _ in range(runs):
                    if STATE.stop_requested:
                        return
                    self.run_single(model, command)
                    if STATE.stop_requested:
                        return
                    self.reset_after_completion()
        finally:
            with STATE.lock:
                STATE.benchmark_active = False
                if STATE.phase != MissionPhase.STOPPED:
                    STATE.phase = MissionPhase.IDLE

    def run_single(self, model: str, command: str) -> RunMetrics:
        if not ROS_NODE or not NAVIGATOR:
            raise RuntimeError("ROS node is not ready")
        if model not in Config.SUPPORTED_MODELS:
            raise RuntimeError(f"unsupported model {model}")

        with STATE.lock:
            STATE.stop_requested = False
            STATE.command = command
            STATE.model = model
            STATE.phase = MissionPhase.DISCOVERING
            STATE.actual_route = []
            STATE.heatmap = {}
            STATE.generated_route = []
            STATE.llm_reasoning = ""
            STATE.current_target = "DISCOVERY"
            STATE.run_counters[model] += 1
            run_number = STATE.run_counters[model]

        NAVIGATOR.reset_buffers()
        sampler = MetricsSampler()
        sampler.start()
        start_time = time.perf_counter()
        start_collisions = ROS_NODE.collision_count
        result = RunMetrics(
            run_id=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model.replace(':', '_')}_{uuid.uuid4().hex[:8]}",
            model=model,
            run_number=run_number,
            command=command,
            started_at=now_iso(),
        )

        try:
            if not self._ensure_home_before_run():
                raise RuntimeError("could not verify HOME before starting run")
            # Fixed Gazebo board: install known confirmed cylinders immediately.
            # No exploration gate; this prevents endless spinning and makes every LLM start from the same map.
            self._discover_cylinders()

            with STATE.lock:
                STATE.phase = MissionPhase.PLANNING
                STATE.current_target = "LLM"
            cylinders = CYLINDERS.discovered()
            # If user only asks to go HOME/reset, do it without forcing LLM planning.
            if self._is_home_command(command):
                result.llm_latency = 0.0
                result.route = []
                result.llm_reasoning = "Manual HOME command: robot returned to HOME without benchmark LLM scoring."
                result.return_home_success = NAVIGATOR.return_home()
                result.success = result.return_home_success
                with STATE.lock:
                    STATE.generated_route = []
                    STATE.llm_reasoning = result.llm_reasoning
                    STATE.phase = MissionPhase.COMPLETE if result.success else MissionPhase.FAILED
                return result
            plan = PLANNER.plan(model, command, cylinders, ROS_NODE.get_pose())
            result.llm_latency = float(plan["latency"])
            result.route = list(plan["route"])
            result.llm_reasoning = str(plan["reasoning"])
            with STATE.lock:
                STATE.generated_route = result.route
                STATE.llm_reasoning = result.llm_reasoning

            waypoints = []
            for label in result.route:
                cyl = CYLINDERS.by_label(label)
                if cyl:
                    waypoints.append(NAVIGATOR.inspection_pose_for(cyl))
            with STATE.lock:
                STATE.planned_route_xy = [(Config.HOME_X, Config.HOME_Y)] + waypoints + [(Config.HOME_X, Config.HOME_Y)]

            with STATE.lock:
                STATE.phase = MissionPhase.RUNNING
            nav_start = time.perf_counter()
            ok = NAVIGATOR.follow_route(waypoints, result.route)
            result.navigation_time = time.perf_counter() - nav_start
            if not ok:
                result.failure_reason = "route execution failed or timed out"
                STATE.log(result.failure_reason + " — saving run and attempting HOME, not aborting benchmark.", "warning")

            result.return_home_success = NAVIGATOR.return_home()
            # For data collection, a run is successful only when the route executed,
            # collisions stayed low, and the robot returned home. Failures are saved
            # as FAIL, not ABORTED, so the benchmark continues to the next run/model.
            result.success = bool(ok and result.return_home_success and (ROS_NODE.collision_count - start_collisions) <= 2)
            if not result.success and not result.failure_reason:
                if not result.return_home_success:
                    result.failure_reason = "mission completed but return-home verification failed"
                elif (ROS_NODE.collision_count - start_collisions) > 2:
                    result.failure_reason = "too many collision proximity events"
                else:
                    result.failure_reason = "mission did not satisfy success criteria"
            with STATE.lock:
                STATE.phase = MissionPhase.COMPLETE if result.success else MissionPhase.FAILED
        except Exception as exc:
            result.failure_reason = str(exc)
            result.return_home_success = NAVIGATOR.return_home() if not STATE.stop_requested else False
            result.success = False
            with STATE.lock:
                # Only user stop is ABORT/STOPPED. Normal system failures are FAIL
                # and still saved so data collection can continue.
                STATE.phase = MissionPhase.STOPPED if STATE.stop_requested else MissionPhase.FAILED
                STATE.last_failure = result.failure_reason
        finally:
            ROS_NODE.stop_robot()
            cpu, ram, power = sampler.stop()
            result.cpu_avg = cpu
            result.ram_avg = ram
            result.power_avg = power
            result.completed_at = now_iso()
            result.mission_time = time.perf_counter() - start_time
            result.collisions = max(0, ROS_NODE.collision_count - start_collisions)
            result.replan_count = NAVIGATOR.replan_count
            with STATE.lock:
                result.trajectory = [{"x": x, "y": y} for x, y in STATE.actual_route]
                result.distance = path_length(STATE.actual_route)
                STATE.results.append(result)
            DataStore.append(result)
            STATE.log(f"Run {result.run_number} for {model} saved: success={result.success} home={result.return_home_success} fail={result.failure_reason or 'none'}", "info")
        return result

    def _discover_cylinders(self) -> bool:
        """Use fixed 9-cylinder Gazebo layout instead of active exploration.

        This is the critical stability change. The robot no longer spins in an
        exploration state waiting for LiDAR confirmation. The LiDAR remains used
        for obstacle avoidance and collision metrics, while the LLM is evaluated
        fairly on route/task planning over the same fixed board every run.
        """
        install_known_cylinders()
        with STATE.lock:
            STATE.phase = MissionPhase.PLANNING
            STATE.current_target = "KNOWN_MAP_READY"
        STATE.log(f"Known Gazebo layout loaded: {len(CYLINDERS.discovered())} cylinders. Exploration skipped.", "info")
        return True

    def _is_home_command(self, command: str) -> bool:
        c = (command or "").strip().lower()
        return c in ("home", "go home", "return home", "reset", "reset home", "go to home") or ("home" in c and "inspect" not in c and "cyl" not in c)

    def _reset_internal_state(self, clear_cylinders: bool) -> None:
        if clear_cylinders:
            CYLINDERS.clear()
        if NAVIGATOR:
            NAVIGATOR.reset_buffers()
        STATE.reset_transient(keep_results=True)


ENGINE = MissionEngine()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def mission_page() -> str:
        return render_template_string(DASHBOARD_HTML, page="mission")

    @app.route("/individual")
    def individual_page() -> str:
        return render_template_string(DASHBOARD_HTML, page="individual")

    @app.route("/comparison")
    def comparison_page() -> str:
        return render_template_string(DASHBOARD_HTML, page="comparison")

    @app.route("/export")
    def export_page() -> str:
        return render_template_string(DASHBOARD_HTML, page="export")

    @app.route("/camera")
    def camera_page() -> str:
        return render_template_string(DASHBOARD_HTML, page="camera")

    @app.route("/map")
    def map_page() -> str:
        return render_template_string(DASHBOARD_HTML, page="map")

    @app.route("/logs")
    def logs_page() -> str:
        return render_template_string(DASHBOARD_HTML, page="logs")

    @app.route("/video_feed")
    def video_feed() -> Response:
        def stream() -> Iterable[bytes]:
            while True:
                frame = CAMERA.get_jpeg()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.1)
        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def api_status() -> Response:
        with STATE.lock:
            state = {
                "phase": STATE.phase.value,
                "model": STATE.model,
                "command": STATE.command,
                "llm_reasoning": STATE.llm_reasoning,
                "generated_route": STATE.generated_route,
                "current_target": STATE.current_target,
                "planned_route": STATE.planned_route_xy,
                "actual_route": STATE.actual_route[-2000:],
                "heatmap": STATE.heatmap,
                "home": [Config.HOME_X, Config.HOME_Y],
                "logs": STATE.logs[-120:],
                "stop_requested": STATE.stop_requested,
                "benchmark_active": STATE.benchmark_active,
                "model_switching": ENGINE.model_switching,
                "home_required_before_run": STATE.home_required_before_run,
                "last_failure": STATE.last_failure,
            }
        state["cylinders"] = [asdict(c) for c in CYLINDERS.all()]
        state["system"] = system_status()
        state["camera"] = CAMERA.status()
        state["ros"] = ROS_NODE.ros_status() if ROS_NODE else {"available": False}
        state["ollama"] = PLANNER.status()
        state["analysis"] = aggregate_results(STATE.results)
        return jsonify(state)

    @app.route("/api/mission", methods=["POST"])
    def api_mission() -> Response:
        payload = request.get_json(force=True, silent=True) or {}
        command = str(payload.get("command") or "Inspect all cylinders and return home.")
        model = str(payload.get("model") or STATE.model)
        ok = ENGINE.start_single(model, command)
        return jsonify({"ok": ok, "message": "mission started" if ok else "controller is busy"})

    @app.route("/api/start_benchmark", methods=["POST"])
    def api_start_benchmark() -> Response:
        payload = request.get_json(force=True, silent=True) or {}
        command = str(payload.get("command") or "Inspect all cylinders and return home.")
        requested = payload.get("models") or Config.SUPPORTED_MODELS
        models = [m for m in requested if m in Config.SUPPORTED_MODELS]
        runs = max(1, min(999, int(payload.get("runs") or Config.RUNS_PER_MODEL)))
        ok = ENGINE.start_benchmark(models, runs, command)
        return jsonify({"ok": ok, "models": models, "runs": runs})

    @app.route("/api/stop", methods=["POST"])
    def api_stop() -> Response:
        ENGINE.stop()
        return jsonify({"ok": True, "message": "stopped without forced reset"})

    @app.route("/api/resume", methods=["POST"])
    def api_resume() -> Response:
        ENGINE.resume()
        return jsonify({"ok": True})

    @app.route("/api/return_home", methods=["POST"])
    def api_return_home() -> Response:
        ok = ENGINE.return_home()
        return jsonify({"ok": ok})

    @app.route("/api/reset", methods=["POST"])
    def api_reset() -> Response:
        ok = ENGINE.reset_after_completion()
        return jsonify({"ok": ok})

    @app.route("/api/model", methods=["POST"])
    def api_model() -> Response:
        payload = request.get_json(force=True, silent=True) or {}
        ok, msg = ENGINE.switch_model(str(payload.get("model") or ""))
        return jsonify({"ok": ok, "message": msg, "model": STATE.model})

    @app.route("/api/cylinders")
    def api_cylinders() -> Response:
        return jsonify([asdict(c) for c in CYLINDERS.all()])

    @app.route("/api/results")
    def api_results() -> Response:
        return jsonify([asdict(r) for r in STATE.results])

    @app.route("/api/analysis")
    def api_analysis() -> Response:
        return jsonify(aggregate_results(STATE.results))

    @app.route("/api/comparison")
    def api_comparison() -> Response:
        return jsonify(aggregate_results(STATE.results))

    @app.route("/export_thesis_csv")
    def export_csv() -> Response:
        ensure_csv_exists()
        return send_file(Config.CSV_PATH, as_attachment=True, download_name="openclaw_benchmark_runs.csv")

    @app.route("/export_combined_json")
    def export_json() -> Response:
        DataStore._rewrite_all_json()
        return send_file(Config.JSON_PATH, as_attachment=True, download_name="openclaw_benchmark_runs.json")

    @app.route("/export_report")
    def export_report() -> Response:
        path = generate_report()
        return send_file(path, as_attachment=True, download_name=os.path.basename(path))

    return app


def system_status() -> Dict[str, Any]:
    cpu, ram, power = read_system_metrics()
    return {"cpu": cpu, "ram": ram, "power": power}


def ensure_csv_exists() -> None:
    if os.path.exists(Config.CSV_PATH):
        return
    with open(Config.CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=DataStore.CSV_FIELDS).writeheader()


def generate_report() -> str:
    stats = aggregate_results(STATE.results)
    path = os.path.join(Config.REPORT_DIR, f"openclaw_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report = {
        "title": "Evaluation of a Latency-Aware LLM-Based Framework for Intelligent Task Planning and Autonomous Navigation in ROS 2 Mobile Robots",
        "generated_at": now_iso(),
        "supported_models": Config.SUPPORTED_MODELS,
        "run_count": len(STATE.results),
        "statistics": stats,
        "results": [asdict(r) for r in STATE.results],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenClaw Benchmark</title>
  <style>
    :root { --bg:#101418; --panel:#171d24; --panel2:#1d2530; --text:#eef3f8; --muted:#9dadbd; --line:#2c3642; --accent:#43b3a9; --warn:#e6b450; --bad:#ef6f6c; --good:#66c27a; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif; }
    header { height:58px; display:flex; align-items:center; justify-content:space-between; padding:0 18px; border-bottom:1px solid var(--line); background:#121820; position:sticky; top:0; z-index:5; }
    h1 { font-size:18px; margin:0; font-weight:700; }
    nav { display:flex; gap:8px; }
    nav a, button, select, input { border:1px solid var(--line); background:var(--panel2); color:var(--text); border-radius:6px; padding:8px 10px; text-decoration:none; }
    button { cursor:pointer; }
    button.primary { background:var(--accent); border-color:var(--accent); color:#061213; font-weight:700; }
    button.danger { background:var(--bad); border-color:var(--bad); color:#160303; font-weight:700; }
    main { display:grid; grid-template-columns:360px minmax(420px,1fr) 360px; gap:12px; padding:12px; min-height:calc(100vh - 58px); }
    section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; min-width:0; }
    h2 { font-size:14px; margin:0 0 10px; color:#dfe8f0; }
    textarea { width:100%; min-height:86px; resize:vertical; border:1px solid var(--line); background:#101820; color:var(--text); border-radius:6px; padding:10px; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
    .stack { display:grid; gap:12px; }
    .metric { display:grid; grid-template-columns:1fr auto; gap:8px; padding:7px 0; border-bottom:1px solid #26313d; }
    .muted { color:var(--muted); }
    .camera { width:100%; aspect-ratio:16/9; object-fit:cover; background:#0c1015; border-radius:6px; border:1px solid var(--line); }
    canvas.map { width:100%; height:520px; background:#0d1218; border:1px solid var(--line); border-radius:6px; }
    .logs { height:280px; overflow:auto; font-family:ui-monospace,Consolas,monospace; font-size:12px; background:#0d1218; border:1px solid var(--line); border-radius:6px; padding:8px; }
    .pill { display:inline-flex; align-items:center; gap:5px; padding:4px 8px; border-radius:999px; background:#222d38; color:#dfe8f0; }
    .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { padding:7px; border-bottom:1px solid var(--line); text-align:left; }
    @media (max-width:1100px) { main { grid-template-columns:1fr; } canvas.map { height:420px; } }
  </style>
</head>
<body>
  <header>
    <h1>OpenClaw LLM Navigation Benchmark</h1>
    <nav>
      <a href="/">Control</a><a href="/camera">Camera</a><a href="/map">Map</a><a href="/logs">Logs</a><a href="/individual">Individual</a><a href="/comparison">Comparison</a><a href="/export">Export</a>
    </nav>
  </header>
  <main>
    <div class="stack">
      <section>
        <h2>Mission Chat</h2>
        <textarea id="command">Inspect all cylinders and return home.</textarea>
        <div class="row" style="margin-top:8px">
          <select id="model"></select>
          <button class="primary" onclick="startMission()">Run</button>
          <input id="runs" type="number" min="1" max="999" value="5" style="width:76px"><button onclick="startBenchmark()">Run benchmark</button>
          <button class="danger" onclick="stopMission()">Stop</button>
          <button onclick="returnHome()">Home</button>
        </div>
      </section>
      <section>
        <h2>LLM Reasoning</h2>
        <div id="reasoning" class="muted"></div>
        <div style="margin-top:10px"><span class="pill" id="route">route pending</span></div>
      </section>
      <section>
        <h2>Mission Logs</h2>
        <div id="logs" class="logs"></div>
      </section>
    </div>
    <div class="stack">
      <section>
        <h2>Live Arena Map</h2>
        <canvas id="map" class="map" width="900" height="620"></canvas>
      </section>
      <section>
        <h2>Trajectory Heatmap</h2>
        <canvas id="heatmap" class="map" width="900" height="360"></canvas>
      </section>
    </div>
    <div class="stack">
      <section>
        <h2>Live Gazebo Camera</h2>
        <img class="camera" src="/video_feed">
      </section>
      <section>
        <h2>System Monitor</h2>
        <div id="monitor"></div>
      </section>
      <section>
        <h2>Model Analysis</h2>
        <canvas id="chart" height="210"></canvas>
      </section>
      <section>
        <h2>Recent Runs</h2>
        <div id="runs"></div>
      </section>
    </div>
  </main>
<script>
const models = ["gemma2:2b","qwen2.5:3b","deepseek-r1:1.5b"];
const modelSelect = document.getElementById("model");
models.forEach(m => { const o=document.createElement("option"); o.value=m; o.textContent=m; modelSelect.appendChild(o); });
let state = null, selectedModel = models[0], switchingModel = false;
async function post(url, body={}) { return fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json()); }
function startMission(){ selectedModel=modelSelect.value; post("/api/mission",{model:selectedModel, command:document.getElementById("command").value}); }
function startBenchmark(){ const runs=Math.max(1,Math.min(999,parseInt(document.getElementById("runs").value||"5"))); post("/api/start_benchmark",{models, runs, command:document.getElementById("command").value}); }
function stopMission(){ post("/api/stop"); }
function returnHome(){ post("/api/return_home"); }
modelSelect.addEventListener("change", async () => {
  selectedModel=modelSelect.value; switchingModel=true;
  try {
    const res = await post("/api/model",{model:selectedModel});
    if(!res.ok){ alert(res.message || "Model change failed"); selectedModel=state?.model || models[0]; }
  }
  finally { switchingModel=false; }
});
function tx(x,w){ return w/2 + x*70; } function ty(y,h){ return h/2 - y*70; }
function drawMap() {
  const c=document.getElementById("map"), ctx=c.getContext("2d"), w=c.width, h=c.height; ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#26323d"; ctx.lineWidth=1; for(let x=0;x<w;x+=70){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();} for(let y=0;y<h;y+=70){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  if(!state) return;
  const path=state.actual_route||[]; ctx.strokeStyle="#43b3a9"; ctx.lineWidth=3; ctx.beginPath(); path.forEach((p,i)=>{ const x=tx(p[0],w), y=ty(p[1],h); if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);}); ctx.stroke();
  const plan=state.planned_route||[]; ctx.setLineDash([8,8]); ctx.strokeStyle="#e6b450"; ctx.lineWidth=2; ctx.beginPath(); plan.forEach((p,i)=>{ const x=tx(p[0],w), y=ty(p[1],h); if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);}); ctx.stroke(); ctx.setLineDash([]);
  const home=state.home||[0,0]; ctx.save(); ctx.globalAlpha=0.28; ctx.fillStyle="#49d17d"; ctx.beginPath(); ctx.arc(tx(home[0],w),ty(home[1],h),28,0,Math.PI*2); ctx.fill(); ctx.globalAlpha=1; ctx.strokeStyle="#49d17d"; ctx.lineWidth=3; ctx.stroke(); ctx.fillStyle="#b8ffd0"; ctx.font="bold 15px system-ui"; ctx.fillText("HOME",tx(home[0],w)-24,ty(home[1],h)+5); ctx.restore();
  (state.cylinders||[]).forEach(cy=>{ const x=tx(cy.x,w), y=ty(cy.y,h); ctx.fillStyle=cy.discovered?"#6aa9ff":"#607080"; ctx.beginPath(); ctx.arc(x,y,12,0,Math.PI*2); ctx.fill(); ctx.strokeStyle="#ffffff"; ctx.lineWidth=2; ctx.stroke(); ctx.fillStyle="rgba(0,0,0,.72)"; ctx.fillRect(x-23,y-34,52,20); ctx.fillStyle="#ffffff"; ctx.font="bold 12px system-ui"; ctx.fillText(cy.label,x-20,y-20); });
  const pose=state.ros&&state.ros.pose; if(pose){ const x=tx(pose.x,w), y=ty(pose.y,h); ctx.fillStyle="#f2f5f7"; ctx.beginPath(); ctx.arc(x,y,8,0,Math.PI*2); ctx.fill(); ctx.strokeStyle="#f2f5f7"; ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x+22*Math.cos(pose.yaw),y-22*Math.sin(pose.yaw)); ctx.stroke(); }
}
function drawHeatmap(){
  const c=document.getElementById("heatmap"), ctx=c.getContext("2d"), w=c.width,h=c.height; ctx.clearRect(0,0,w,h); if(!state)return;
  const vals=Object.entries(state.heatmap||{}); const max=Math.max(1,...vals.map(v=>v[1]));
  vals.forEach(([k,v])=>{ const [x,y]=k.split(",").map(Number); const a=v/max; ctx.fillStyle=`rgba(239,111,108,${0.12+0.75*a})`; ctx.beginPath(); ctx.arc(tx(x,w),ty(y,h),5+12*a,0,Math.PI*2); ctx.fill(); });
}
function metric(label,value){ return `<div class="metric"><span class="muted">${label}</span><strong>${value}</strong></div>`; }
function refreshChart(analysis){
  const c=document.getElementById("chart"), ctx=c.getContext("2d"), w=c.width=c.clientWidth*devicePixelRatio, h=c.height=260*devicePixelRatio;
  ctx.scale(devicePixelRatio,devicePixelRatio); const cw=c.clientWidth, ch=260; ctx.clearRect(0,0,cw,ch);
  const colors={"gemma2:2b":"#ef6f6c","qwen2.5:3b":"#6aa9ff","deepseek-r1:1.5b":"#66c27a"};
  const metrics=[
    {name:"Latency", unit:"s", values:models.map(m=>analysis[m]?.llm_latency?.avg||0)},
    {name:"Mission", unit:"s", values:models.map(m=>analysis[m]?.mission_time?.avg||0)},
    {name:"Success", unit:"%", values:models.map(m=>analysis[m]?.success_rate||0), max:100},
    {name:"Collision", unit:"", values:models.map(m=>analysis[m]?.collisions?.avg||0)},
    {name:"Power", unit:"W", values:models.map(m=>analysis[m]?.power?.avg||0)}
  ];
  const panelW=cw/metrics.length, base=220;
  ctx.font="11px system-ui"; ctx.fillStyle="#9dadbd";
  metrics.forEach((metric,mi)=>{
    const x0=mi*panelW+8, max=Math.max(metric.max||1,...metric.values);
    ctx.fillStyle="#9dadbd"; ctx.fillText(metric.name,x0,14);
    metric.values.forEach((v,i)=>{
      const bh=(base-42)*(v/max), bw=Math.max(9,(panelW-28)/4), x=x0+8+i*(bw+5), y=base-bh;
      ctx.fillStyle=colors[models[i]]; ctx.fillRect(x,y,bw,bh);
      ctx.fillStyle="#dfe8f0"; ctx.fillText((v||0).toFixed(metric.name==="Success"?0:1),x,Math.max(28,y-4));
    });
  });
  models.forEach((m,i)=>{ ctx.fillStyle=colors[m]; ctx.fillRect(10+i*112,238,10,10); ctx.fillStyle="#9dadbd"; ctx.fillText(m,24+i*112,248); });
}
async function tick(){
  state = await fetch("/api/status").then(r=>r.json());
  if(!switchingModel && document.activeElement!==modelSelect){ selectedModel=state.model; }
  modelSelect.value=selectedModel;
  document.getElementById("reasoning").textContent=state.llm_reasoning||"Waiting for planning output.";
  document.getElementById("route").textContent=(state.generated_route||[]).join(" -> ") || "route pending";
  document.getElementById("logs").innerHTML=(state.logs||[]).map(l=>`<div><span class="muted">${l.time}</span> ${l.level}: ${l.message}</div>`).join("");
  const s=state.system, ros=state.ros, cam=state.camera, oll=state.ollama;
  document.getElementById("monitor").innerHTML =
    metric("Phase",state.phase)+metric("Selected model",state.model)+(state.home_required_before_run?metric("Run precheck","HOME required"):"")+metric("CPU",`${s.cpu.toFixed(1)}%`)+metric("RAM",`${s.ram.toFixed(1)}%`)+metric("Jetson power",`${s.power.toFixed(2)} W`)+
    metric("ROS",ros.available?`odom ${ros.odom_age?.toFixed(1)??"-"}s scan ${ros.scan_age?.toFixed(1)??"-"}s`:"offline")+metric("Ollama",oll.online?"online":"offline")+metric("Camera",cam.available?"Gazebo stream":"unavailable");
  drawMap(); drawHeatmap(); refreshChart(state.analysis||{});
  const results=await fetch("/api/results").then(r=>r.json());
  document.getElementById("runs").innerHTML=`<table><tr><th>Model</th><th>Run</th><th>OK</th><th>Latency</th></tr>`+results.slice(-8).reverse().map(r=>`<tr><td>${r.model}</td><td>${r.run_number}</td><td>${r.success}</td><td>${r.llm_latency.toFixed(2)}</td></tr>`).join("")+`</table>`;
}
setInterval(tick,1000); tick();
</script>
</body>
</html>
"""


def start_ros() -> None:
    global ROS_NODE, NAVIGATOR
    if rclpy is None:
        raise RuntimeError("ROS 2 Python packages are not available. Run this on the Jetson ROS 2 Humble environment.")
    rclpy.init()
    ROS_NODE = OpenClawRosNode()
    NAVIGATOR = SmoothNavigator(ROS_NODE)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(ROS_NODE)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    STATE.log("ROS node started.", "info")


def load_previous_results() -> None:
    loaded = DataStore.load_existing()
    with STATE.lock:
        STATE.results = loaded
        for result in loaded:
            if result.model in STATE.run_counters:
                STATE.run_counters[result.model] = max(STATE.run_counters[result.model], result.run_number)
    ensure_csv_exists()


def main() -> None:
    load_previous_results()
    start_ros()
    app = create_app()
    STATE.log(f"Dashboard ready on http://{Config.FLASK_HOST}:{Config.FLASK_PORT}", "info")
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
