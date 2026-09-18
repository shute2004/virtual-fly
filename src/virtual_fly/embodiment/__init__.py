"""Reusable embodied runtime for production virtual-fly experiments.

CLI entry points live under ``scripts/``; scientific/runtime implementation lives here.
"""

from .course import CourseEvent, CourseObservation, FlyppyCourse, Gate
from .haltere import HaltereCampaniformSensor, HaltereSensorySnapshot
from .periphery import WholeBodyPeriphery, WholeBodyPeripheralSnapshot
from .retina import MaleCNSRetina, RetinalDrive
from .world import FlyppyWorld

__all__ = [
    "CourseEvent", "CourseObservation", "FlyppyCourse", "Gate",
    "FlyppyWorld", "MaleCNSRetina", "RetinalDrive",
    "HaltereCampaniformSensor", "HaltereSensorySnapshot",
    "WholeBodyPeriphery", "WholeBodyPeripheralSnapshot",
]
