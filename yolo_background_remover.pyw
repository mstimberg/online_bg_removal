"""
Script that watches a folder of files, removes the background and saves them as compressed files (optionally in a new
folder). The input file names need to have filenames ending in consecutive numbers.
"""

import concurrent
import glob
import gzip
import io
import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from multiprocessing import JoinableQueue, set_start_method

import imageio
import imageio.plugins.ffmpeg as ffmpeg_plugin
import numpy as np
import pandas as pd
import psutil
import pyqtgraph as pg
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtWidgets
import tifffile
import torch
import yaml
from pyqtgraph import RectROI
from PySide6.QtCore import QLocale, Qt
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from image_processing.trajectory_analysis import (
    calculate_features,
    mark_avoiding_reactions_from_motion,
    segments_from_table,
)

pg.setConfigOptions(imageAxisOrder="row-major")

DEFAULT_MIN_AREA = 400
DEFAULT_MAX_AREA = 12500

MASK_COLOR = (255, 195, 0)
PREVIEW_EVERY = 100  # Show a preview image every X frames

MAX_FILE_READ_THREADS = 4

FFMPEG_VCODEC = "libx264"
FFMPEG_PRESET = "fast"
FFMPEG_PIX_FMT = "yuv420p"
CONSOLE_LOG_LEVEL = logging.WARNING
FILE_LOG_LEVEL = logging.INFO

MAX_FFMPEG_PROCESSES = 4  # Maximum number of ffmpeg processes to run in parallel

FRAMES_PER_ZIP = 1_000_000  # a single zip file...

# None: Use CUDA, if available, otherwise CPU.
# Set to "mps" for MPS on MacOS
DEVICE = None

# Class to store settings, docs, type, and min/max (or options) for each parameter
@dataclass
class Setting:
    """Class to store settings, docs, type, and min/max (or options) for each parameter"""
    name: str
    doc: str
    type: type
    unit: str|None = None
    str_evaluate: bool = False
    zero_as_none: bool = False  # If True, 0 is treated as None
    min: float|None = None
    max: float|None = None
    options: list|None = None
    default: object = None


DEFAULT_TRACK_SETTINGS = {
    "package": "trackpy",
    "packages": {
        "trackpy": {
            "maximum_speed": Setting(
                name="maximum_speed",
                doc="The maximum speed at which features can move (will be translated into a maximum pixel distance between frames)",
                type=float,
                unit="µm/s",
                default=3000,
                min=0,
                max=10000,
            ),
            "memory": Setting(
                name="memory",
                doc="The maximum time during which a feature can vanish, then reappear nearby (will be converted into frames)",
                unit="s",
                default=0.5,
                type=float,
                min=0,
                max=10,
            ),
            "adaptive_stop": Setting(
                name="adaptive_stop",
                doc="If not None, when encountering an oversize subnet, retry by progressively reducing maximum_speed until the subnet is solvable. If maximum_speed becomes <= adaptive_stop, give up and raise a SubnetOversizeException.",
                zero_as_none=True,
                min=0,
                max=10000,
                unit="µm/s",
                default=0,
                type=float,
            ),
            "adaptive_step": Setting(
                name="adaptive_step",
                doc="Reduce search_range by multiplying it by this factor.",
                default=0.95,
                min=0,
                max=1.0,
                type=float,
            ),
            "neighbor_strategy": Setting(
                name="neighbor_strategy",
                doc="Algorithm used to identify nearby features. Default 'KDTree'.",
                type=str,
                default="KDTree",
                options=["KDTree", "BTree"],
            ),
            "link_strategy": Setting(
                name="link_strategy",
                doc="Algorithm used to resolve subnetworks of nearby particles. 'auto' uses hybrid (numba+recursive) if available. 'drop' causes particles in subnetworks to go unlinked.",
                type=str,
                default="auto",
                options=[
                    "recursive",
                    "nonrecursive",
                    "numba",
                    "hybrid",
                    "drop",
                    "auto",
                ],
            ),
        },
    }
}

DEFAULT_TRACK_SETTINGS["packages"]["norfair"] = {
    "maximum_speed": DEFAULT_TRACK_SETTINGS["packages"]["trackpy"]["maximum_speed"],
    "memory": DEFAULT_TRACK_SETTINGS["packages"]["trackpy"]["memory"],
    "initialization_delay": Setting(
        name="initialization_delay",
        doc="The time to wait before starting to track. It must be smaller than `memory` or otherwise the object would never be initialized.",
        type=float,
        min=0,
        max=10,
        unit="s",
        default=0.2,
    ),
}

DEFAULT_TRACK_SETTINGS["packages"]["yolo"] = {
    "tracker_type": Setting(
        name="tracker_type",
        doc="The tracker to use",
        type=str,
        options=[
            "botsort",
            "bytetrack",
            "ocsort",
            "deepocsort",
            "fasttrack",
            "tracktrack",
        ],
    )
}


# Convenience class to build simple GUI for settings
class SettingGUI(QtWidgets.QWidget):
    def __init__(self, settings, setting_values, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setting_values = setting_values
        self.layout = QtWidgets.QFormLayout()
        for name, setting in settings.items():
            value = setting_values.get(name, setting.default)
            widget = self.create_widget(setting, value)
            self.layout.addRow(setting.name, widget)
        self.setLayout(self.layout)

    def create_widget(self, setting, value):
        if setting.options:
            widget = QtWidgets.QComboBox()
            for option in setting.options:
                widget.addItem(option)
            widget.setCurrentText(value)
        elif setting.type == bool:
            widget = QtWidgets.QCheckBox()
            widget.setChecked(value)
        elif setting.type == int:
            widget = QtWidgets.QSpinBox()
            widget.setMinimum(setting.min)
            widget.setMaximum(setting.max)
            widget.setValue(value)
        elif setting.type == float:
            widget = QtWidgets.QDoubleSpinBox()
            widget.setMinimum(setting.min)
            widget.setMaximum(setting.max)
            widget.setValue(value)
            if setting.unit:
                widget.setSuffix(f" {setting.unit}")
        else:
            raise NotImplementedError(f"Type {setting.type} not implemented")
        widget.setToolTip(setting.doc)
        return widget

    def get_settings(self):
        settings = {}
        for i, setting in zip(range(self.layout.rowCount()), self.settings):
            widget = self.layout.itemAt(i, QtWidgets.QFormLayout.FieldRole).widget()
            if isinstance(widget, QtWidgets.QComboBox):
                settings[setting] = widget.currentText()
            elif isinstance(widget, QtWidgets.QCheckBox):
                settings[setting] = widget.isChecked()
            else:
                settings[setting] = widget.value()
        return settings

TRACE = False
if TRACE:
    from viztracer import get_tracer

set_start_method("spawn", force=True)  # Required for multiprocessing with CUDA

# Global variable to track whether a thread raised an exception
exception_occured = False
# Global variable to pause reading of files in case of low memory
pause_reading = False

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(CONSOLE_LOG_LEVEL)
_console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(threadName)s : %(levelname)s : %(message)s")
)
logger.addHandler(_console_handler)


# Make sure that log messages always have an index
def default_index_filter(record: logging.LogRecord):
    record.index = record.index if hasattr(record, "index") else -1
    return record


def human_filesize(num):
    for unit in ("", "K", "M"):
        if abs(num) < 1000:
            return f"{num:.0f} {unit}B"
        num /= 1000
    return f"{num:.0f} GB"


def human_duration(time_in_s):
    time_in_s = int(time_in_s)
    hours = time_in_s // 3600
    time_in_s -= hours * 3600
    minutes = time_in_s // 60
    time_in_s -= minutes * 60

    if hours:
        time_string = f"{hours}h, {minutes}m, {time_in_s}s"
    elif minutes:
        time_string = f"{minutes}m, {time_in_s}s"
    else:
        time_string = f"{time_in_s}s"
    return time_string


def get_free_space(directory):
    if os.path.exists(directory):
        _, _, free = shutil.disk_usage(directory)
        return human_filesize(free)
    else:
        return "?"


def open_dir(directory):
    if platform.system() == "Windows":
        os.startfile(directory)
    else:
        print("Only supported on Windows")


def read_image_tifffile(path):
    return tifffile.imread(path)


def read_image_imageio(path):
    return imageio.v3.imread(path)


def read_image_libtiff(path):
    from libtiff import TIFF

    tiff = TIFF.open(path)
    frame = tiff.read_image()
    tiff.close()
    return frame


def get_image_fnames(dirname):
    return sorted(
        glob.glob(os.path.join(dirname, "*.tiff"))
        + glob.glob(os.path.join(dirname, "*.tif"))
        + glob.glob(os.path.join(dirname, "*.png"))
    )

read_functions = {
    "tifffile": read_image_tifffile,
    "libtiff": read_image_libtiff,
    "imageio": read_image_imageio,
}


def write_image(path, array, pixelsize=None, compression=None):
    if not isinstance(path, str) or path.endswith(".tiff"):
        if pixelsize:
            metadata = {
                "PhysicalSizeX": pixelsize,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixelsize,
                "PhysicalSizeYUnit": "µm",
            }
            resolution = (1e4 / pixelsize, 1e4 / pixelsize)
        else:
            metadata = {}
            resolution = None
        colormap = np.zeros((3, 256), dtype=np.uint16)
        # Fill in the colors that are actually used
        min, max = int(array.min()), int(array.max())
        if min == max:
            colormap[:, min : max + 1] = (
                65535  # all white instead of all black for empty images
            )
        else:
            colormap[:, min : max + 1] = np.linspace(
                0, 65535, max - min + 1, dtype=np.uint16
            )
        imageio.v3.imwrite(
            path,
            array,
            compression=compression,
            extension=".tiff",
            plugin="tifffile",
            metadata=metadata,
            resolution=resolution,
            colormap=colormap,
            resolutionunit="CENTIMETER",
        )
    else:
        # Write with default parameters for PNG (background)
        imageio.v3.imwrite(path, array, compression_level=0)


def divisors(number):
    return [i for i in range(1, number + 1) if number % i == 0]


def get_roi_slice(roi_selector):
    """
    Return ROI slices from the selector for row-major frames.
    """
    x_start = int(round(float(roi_selector.pos()[0])))
    y_start = int(round(float(roi_selector.pos()[1])))
    width = int(round(float(roi_selector.size()[0])))
    height = int(round(float(roi_selector.size()[1])))

    x_stop = x_start + width
    y_stop = y_start + height
    return slice(y_start, y_stop), slice(x_start, x_stop)


def extract_file_number(filename):
    try:
        name = os.path.splitext(filename)[0]
        file_number = re.search(r"\d+$", name)
        file_number = int(file_number.group()) if file_number else None
    except (ValueError, TypeError):
        file_number = None
    return file_number


def filename_prefix(fnames, prefix=True):
    """
    Get a file name prefix from a list of file names. If `prefix=True`, assumes
    that the name will be used as a prefix (followed by numbers), leaving characters
    such as `_` or `-` in the end in place – with `prefix=False`, those will be removed.
    """
    file_name = os.path.commonprefix([os.path.basename(f) for f in fnames])
    if prefix:
        to_strip = "0 "
    else:
        to_strip = "0 _-:$"
    file_name = file_name.rstrip(to_strip)  # Remove unuseful characters at the end
    return file_name

def verify_tiff(fname : str) -> bool:
    """
    Quick check whether a TIFF exists and is not truncated. This is not a formal validation
    of the actual content, but only checks for the most common issue of a truncated (not yet
    completely written) file.
    """
    if not os.path.exists(fname) or not os.path.isfile(fname):
        return False
    # Read in TIFF header
    try:
        length = os.stat(fname).st_size
        if length < 8: # Not even enough for a minimal TIFF header
            return False
        with open(fname, "rb") as f:
            # Read byte order
            bo = f.read(2)
            if bo == b"II":
                byteorder = "little"
            elif bo == b"MM":
                byteorder = "big"
            else:
                return False
            # Skip 2 bytes for version
            f.read(2)
            # Read offset to first IFD
            offset = int.from_bytes(f.read(4), byteorder)
            if offset + 2 >= length:
                return False
            f.seek(offset)
            # Read number of entries
            entries = int.from_bytes(f.read(2), byteorder=byteorder)
            if offset + 2 + 12 * entries + 4 > length:  # At end 4 zero bytes for no next IFD
                return False
            imageLength = None
            rowsPerStrip = None
            stripOffsets = None
            offsetType = None
            StripByteCounts = None
            byteCountType = None
            for _ in range(entries):
                tag = int.from_bytes(f.read(2), byteorder=byteorder)
                value_type = int.from_bytes(f.read(2), byteorder=byteorder)
                _ = int.from_bytes(f.read(4), byteorder=byteorder)
                # We are only interested in imageLength and rowsPerStrip
                if tag == 257:
                    imageLength = int.from_bytes(f.read(4), byteorder=byteorder)
                elif tag == 278:
                    rowsPerStrip = int.from_bytes(f.read(4), byteorder=byteorder)
                elif tag == 273:
                    stripOffsets = int.from_bytes(f.read(4), byteorder=byteorder)
                    offsetType = value_type
                elif tag == 279:
                    stripByteCounts = int.from_bytes(f.read(4), byteorder=byteorder)
                    byteCountType = value_type
                else:
                    f.read(4)  # Skip value
            if imageLength is None or rowsPerStrip is None or stripOffsets is None or StripByteCounts is None:
                return True  # Non-standard TIFF, but not necessarily invalid
            stripsPerImage = (imageLength + rowsPerStrip - 1) // rowsPerStrip
            strips = []
            stripCounts = []
            if offsetType == 4:  # LONG
                f.seek(stripOffsets)
                for _ in range(stripsPerImage):
                    strips.append(int.from_bytes(f.read(4), byteorder=byteorder))
            else:  # SHORT                
                for _ in range(stripsPerImage):
                    stripOffsets.append(int.from_bytes(f.read(4), byteorder=byteorder))
            if byteCountType == 4:  # LONG
                f.seek(stripByteCounts)
                for _ in range(stripsPerImage):
                    stripCounts.append(int.from_bytes(f.read(4), byteorder=byteorder))
            else:  # SHORT
                for _ in range(stripsPerImage):
                    stripCounts.append(int.from_bytes(f.read(4), byteorder=byteorder))
            for offset, count in zip(strips, stripCounts):
                if offset + count > length:
                    return False

        return True
    except IOError:
        return False


def run_wrapper(func):
    def wrapper(*args, **kwargs):
        global exception_occured
        threading.current_thread().name = QtCore.QThread.currentThread().objectName()
        if TRACE:
            get_tracer().enable_thread_tracing()
        while True:
            logger.info(f"Thread '{threading.current_thread().name}' started")
            try:
                func(*args, **kwargs)
                break  # normal exit
            except Exception:
                logger.exception(
                    f"Error in thread '{threading.current_thread().name}', trying to recover"
                )
                exception_occured = True
        logger.info(f"Thread '{threading.current_thread().name}' finished")
    return wrapper


class FileWatcher(FileSystemEventHandler, QtCore.QObject):
    file_available = QtCore.Signal(str, int, float)

    def __init__(self, dialog, dirname, offset, step):
        super().__init__()
        self.dirname = dirname
        self.offset = offset
        self.step = step
        self.signaled_files = set()
        self.dialog = dialog

    def initial_run(self):
        logger.info("Going through existing files")
        filenames = get_image_fnames(self.dirname)

        for filename in filenames:
            self.handle_file(filename)

    def handle_file(self, filename):
        if filename in self.signaled_files:
            return  # do not signal files twice
        self.signaled_files.add(filename)
        file_number = extract_file_number(filename) - self.offset
        if self.step != 0:
            file_number = (file_number + 1)//self.step
        ctime = os.path.getctime(filename)
        logger.debug(
            f"File '{filename}' became available as {file_number} (ctime: {ctime})",
            extra={"index": file_number},
        )
        # We directly change the last ctime here to avoid the overhead of the event queue
        # The exact timing is not an issue here, since the attributes will only be used for the "auto stop" feature
        # The ctime might already be a bit in the past, so we play it safe by instead using the current time
        self.dialog.last_ctime = time.time()
        self.file_available.emit(filename, file_number, ctime)

    def on_created(self, event):
        filename = event.src_path
        if os.path.splitext(filename)[1] not in ['.tiff', '.tif', '.png']:
            return
        self.handle_file(filename)


class FileReaderThread(QtCore.QThread):
    def __init__(
        self,
        parent,
        pool_size,
        frame_shape,
        read_queue,
        processing_queue,
        write_queue,
        read_function,
        delete_files=False,
    ):
        super().__init__(parent)
        self.file_read_pool = QtCore.QThreadPool()
        self.file_read_pool.setObjectName("FileReaderPool")
        self.file_read_pool.setMaxThreadCount(pool_size)
        self.frame_shape = frame_shape
        self.read_queue = read_queue
        self.processing_queue = processing_queue
        self.write_queue = write_queue
        self.read_function = read_function
        self.delete_files = delete_files

    @run_wrapper
    def run(self):
        while True:
            idx, task = self.read_queue.get()
            if task["type"] == "stop":
                try:
                    logger.info(
                        "Stopping file reader thread, remaining tasks in queue: "
                        + str(self.read_queue.qsize())
                    )
                except NotImplementedError:  # on macOS
                    logger.info("Stopping file reader thread.")
                self.read_queue.task_done(index=idx, measure=False)
                break

            assert task["type"] == "read", task
            fname, epoch, relative_idx = (
                task["fname"],
                task["epoch"],
                task["relative_idx"],
            )
            if task["discard"]:
                logger.debug(f"Discarding file '{fname}' (recording off)")
                # Add file to the task queue to make sure that the background remover keeps track of everything
                self.processing_queue.put(
                    (idx, {"type": "background", "fname": fname, "discard": True})
                )
                if self.delete_files:
                    self.write_queue.put({"type": "delete", "fname": fname, "idx": idx})
                self.read_queue.task_done(index=idx)
                continue

            reader = FileReader(
                fname,
                idx,
                epoch,
                relative_idx=relative_idx,
                frame_shape=self.frame_shape,
                read_queue=self.read_queue,
                task_queue=self.processing_queue,
                file_queue=self.write_queue,
                read_function=self.read_function,
                delete_files=self.delete_files,
            )
            reader.setAutoDelete(True)

            # Hand over the task to the pool if threads are available
            while not self.file_read_pool.tryStart(reader):
                logger.debug("File reader thread pool full, waiting for free thread")
                QtCore.QThread.currentThread().msleep(100)

        # Waiting for the pool tasks to finish
        self.file_read_pool.waitForDone()
        logger.info("File reader thread finished")


class FileReader(QtCore.QRunnable):
    def __init__(
        self,
        filename,
        idx,
        epoch,
        frame_shape,
        relative_idx,
        read_queue,
        task_queue,
        file_queue,
        read_function,
        delete_files,
    ):
        super().__init__()
        self.filename = filename
        self.idx = idx
        self.read_queue = read_queue
        self.task_queue = task_queue
        self.file_queue = file_queue
        self.epoch = epoch
        self.relative_idx = relative_idx
        self.read_function = read_function
        self.delete_files = delete_files
        self.frame_shape = frame_shape  # Only used to create fake frames for failed reads

    def run(self):
        global pause_reading
        while pause_reading:
            logger.debug(
                f"Reading file '{self.filename}' paused", extra={"index": self.idx}
            )
            QtCore.QThread.currentThread().msleep(1000)

        logger.debug(f"Reading file '{self.filename}'", extra={"index": self.idx})
        fail_counter = 0
        while True:
            try:
                if os.path.splitext(self.filename)[1] in ['.tiff', '.tif']:
                    if not verify_tiff(self.filename):
                        raise ValueError("Possibly truncated file")                
                    frame = self.read_function(self.filename)
                else:
                    frame = read_image_imageio(self.filename)
                if frame.size == 0:
                    raise ValueError("Empty frame")
                logger.debug(
                    f"FileReader: read image '{self.filename}'",
                    extra={"index": self.idx},
                )
                self.frame_shape = frame.shape
                break
            except Exception as ex:
                # Give up reading
                if fail_counter > 5:
                    logger.error(
                        f"CORRUPTED FRAME: Loading {self.filename} failed too often, giving up: {str(ex)}",
                    )
                    if self.frame_shape is None:
                        raise IOError("Could not read file, and this is the first frame")

                    # Create a fake empty frame
                    frame = np.ones(self.frame_shape, dtype=np.uint8) * 255
                    break

                # Try again later
                duration = int(2**fail_counter * 100)
                logger.debug(
                    f"Loading {self.filename} failed: {str(ex)}, waiting {duration}ms.."
                )
                QtCore.QThread.currentThread().msleep(duration)
                fail_counter += 1

        self.task_queue.put(
            (
                self.idx,
                {
                    "type": "background",
                    "fname": self.filename,
                    "frame": frame,
                    "epoch": self.epoch,
                    "relative_idx": self.relative_idx,
                    "discard": False,
                },
            )
        )
        if self.delete_files:
            self.file_queue.put(
                {"type": "delete", "fname": self.filename, "idx": self.idx}
            )
        self.read_queue.task_done(index=self.idx)

def create_mask(boxes, frame_shape):
    # Mask everything except for the cells
    mask = np.ones(frame_shape, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = False
    return mask

@torch.compile
def otsu_intraclass_variance(image, thresholds):
    flat_image = image.reshape(-1)
    thresholds = torch.as_tensor(
        thresholds, device=flat_image.device, dtype=flat_image.dtype
    ).reshape(-1)

    if thresholds.numel() == 0:
        return thresholds

    above = flat_image.unsqueeze(0) >= thresholds[:, None]
    above_count = above.sum(dim=1)
    below_count = flat_image.numel() - above_count

    flat_image_2d = flat_image.unsqueeze(0)
    flat_image_sq_2d = flat_image_2d.square()
    above_sum = (above * flat_image_2d).sum(dim=1)
    above_sum_sq = (above * flat_image_sq_2d).sum(dim=1)

    total_sum = flat_image.sum()
    total_sum_sq = flat_image.square().sum()
    below_sum = total_sum - above_sum
    below_sum_sq = total_sum_sq - above_sum_sq

    above_mean = torch.where(above_count > 0, above_sum / above_count.clamp_min(1), 0.)

    below_mean = torch.where(below_count > 0, below_sum / below_count.clamp_min(1), 0.)

    above_var = above_sum_sq / above_count.clamp_min(1) - above_mean.square()
    below_var = below_sum_sq / below_count.clamp_min(1) - below_mean.square()

    total_count = thresholds.new_tensor(float(flat_image.numel()))
    above_weight = above_count.to(dtype=thresholds.dtype) / total_count
    below_weight = below_count.to(dtype=thresholds.dtype) / total_count

    intra_var = above_weight * above_var + below_weight * below_var
    return intra_var.squeeze(0) if intra_var.numel() == 1 else intra_var


@torch.no_grad()
@torch.compile()
def extract_patches_centroid_theta(image, boxes_float):
    """
    image:       (B, H, W) float tensor on GPU, values in [0, 1]
    boxes_float: list/tuple of length B with tensors/arrays of shape (Ni, 4)

    Returns a list (length B) with one dict per frame:
    mask               : (N, Hmax, Wmax) True where patch pixels are valid
    boxes_int          : (N, 4) rounded+clamped integer boxes (xyxy, x2/y2 exclusive)
    masked_image       : (H, W) uint8 image with only box pixels preserved
    centroid_global    : (N, 2) centroid in image coords (x, y)
    orientation        : (N,) orientation angle in radians
    major_axis_length  : (N,) major axis length of the thresholded intensity distribution
    minor_axis_length  : (N,) minor axis length of the thresholded intensity distribution
    """
    device = image.device
    work_dtype = torch.float32 if image.dtype in (torch.float16, torch.bfloat16) else image.dtype
    images = image.to(work_dtype)
    B, H, W = images.shape
    
    outputs = []
    for img, boxes in zip(images, boxes_float):
        if boxes.numel() == 0:
            outputs.append({
                "mask": torch.zeros((0, 0, 0), dtype=torch.bool, device=device),
                "boxes_int": torch.zeros((0, 4), dtype=torch.long, device=device),
                "masked_image": torch.ones((H, W), dtype=torch.uint8, device=device) * 255,
                "centroid_global": torch.zeros((0, 2), dtype=work_dtype, device=device),
                "orientation": torch.zeros((0,), dtype=work_dtype, device=device),
                "major_axis_length": torch.zeros((0,), dtype=work_dtype, device=device),
                "minor_axis_length": torch.zeros((0,), dtype=work_dtype, device=device),
            })
            continue

        N = boxes.shape[0]

        # 1) Round float boxes to integer pixel boxes
        b = torch.round(boxes).to(torch.long)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

        # 2) Clamp to image bounds, enforce at least 1x1 box
        x1 = x1.clamp(0, W - 1)
        y1 = y1.clamp(0, H - 1)
        x2 = x2.clamp(1, W)
        y2 = y2.clamp(1, H)

        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)

        boxes_int = torch.stack([x1, y1, x2, y2], dim=1)

        widths = x2 - x1
        heights = y2 - y1
        Hmax = heights.max().to(int)
        Wmax = widths.max().to(int)

        # 3) Build padded batched patches tensor
        y_grid = torch.arange(Hmax, device=device, dtype=torch.long).view(1, Hmax, 1)   # local y
        x_grid = torch.arange(Wmax, device=device, dtype=torch.long).view(1, 1, Wmax)   # local x

        Y = y1.view(N, 1, 1) + y_grid   # absolute y indices
        X = x1.view(N, 1, 1) + x_grid   # absolute x indices

        mask = (y_grid < heights.view(N, 1, 1)) & (x_grid < widths.view(N, 1, 1))

        # Safe gather indices (masked-out values will be zeroed anyway)
        Yc = Y.clamp(0, H - 1)
        Xc = X.clamp(0, W - 1)

        patches = img[Yc, Xc] * mask.to(work_dtype)

        masked_image = torch.ones_like(img, dtype=work_dtype)
        idx = mask.nonzero(as_tuple=True)   # (n_idx, h_idx, w_idx)
        y_idx = Yc[idx[0], idx[1], 0]       # because Yc is (N,H,1)
        x_idx = Xc[idx[0], 0, idx[2]]       # because Xc is (N,1,W)
        vals = patches[idx]
        masked_image[y_idx, x_idx] = vals

        # 4) Batched local moments (using local patch coordinates)
        y_local = y_grid.to(work_dtype)
        x_local = x_grid.to(work_dtype)

        # Determine a threshold via Otsu's method.

        # Use only valid box pixels to avoid bias from zero padding.
        valid_pixels = patches[mask]
        threshold = compute_otsu_threshold(valid_pixels)
        patches[patches>threshold] = 0.0

        M00 = patches.sum(dim=(1, 2))
        M10 = (patches * y_local).sum(dim=(1, 2))
        M01 = (patches * x_local).sum(dim=(1, 2))
        M11 = (patches * y_local * x_local).sum(dim=(1, 2))
        M20 = (patches * y_local * y_local).sum(dim=(1, 2))
        M02 = (patches * x_local * x_local).sum(dim=(1, 2))

        valid_mass = M00 > 0
        safe_M00 = torch.where(valid_mass, M00, torch.ones_like(M00))
        invM00 = 1.0 / safe_M00
        cy = M10 * invM00
        cx = M01 * invM00

        # If thresholding removes all intensity from a patch, fall back to its box center.
        fallback_cy = (heights.to(work_dtype) - 1) * 0.5
        fallback_cx = (widths.to(work_dtype) - 1) * 0.5
        cy = torch.where(valid_mass, cy, fallback_cy)
        cx = torch.where(valid_mass, cx, fallback_cx)

        mu20 = M20 * invM00 - cy * cy
        mu02 = M02 * invM00 - cx * cx
        mu11 = M11 * invM00 - cy * cx
        mu20 = torch.where(valid_mass, mu20, torch.zeros_like(mu20))
        mu02 = torch.where(valid_mass, mu02, torch.zeros_like(mu02))
        mu11 = torch.where(valid_mass, mu11, torch.zeros_like(mu11))

        covariance = torch.stack(
            [
                torch.stack([mu20, mu11], dim=1),
                torch.stack([mu11, mu02], dim=1),
            ],
            dim=1,
        )
        eigvals, _ = torch.linalg.eigh(covariance)        
        minor_axis_length = 4.0 * torch.sqrt(eigvals[:, 0].clamp_min(0))
        major_axis_length = 4.0 * torch.sqrt(eigvals[:, 1].clamp_min(0))
        
        # Moments are accumulated as (y, x), so the x/y difference is reversed here.
        mu2_diff = mu02 - mu20
        theta = torch.where(
            mu2_diff == 0,
            torch.where(mu11 < 0, -np.pi / 4, np.pi / 4),
            0.5 * torch.arctan2(2 * mu11, mu2_diff),
        )
        theta = torch.where(valid_mass, theta, torch.full_like(theta, float("nan")))
        centroid_x = x1.to(work_dtype) + cx
        centroid_y = y1.to(work_dtype) + cy
        bbox_center_x = (x1.to(work_dtype) + x2.to(work_dtype)) * 0.5
        bbox_center_y = (y1.to(work_dtype) + y2.to(work_dtype)) * 0.5
        centroid_x = torch.where(valid_mass, centroid_x, bbox_center_x)
        centroid_y = torch.where(valid_mass, centroid_y, bbox_center_y)
        centroid_global_xy = torch.stack([centroid_x, centroid_y], dim=1)

        outputs.append({
            "mask": mask,
            "boxes_int": boxes_int,
            "masked_image": masked_image.mul(255).to(torch.uint8),
            "centroid": centroid_global_xy,
            "orientation": theta,
            "major_axis_length": major_axis_length,
            "minor_axis_length": minor_axis_length,
        })

    return outputs

def compute_otsu_threshold(valid_pixels):
    threshold_range = (
        torch.arange(
            (torch.min(valid_pixels * 255) + 1).to(int),
            torch.max(valid_pixels * 255).to(int),
        )
        / 255.0
    )
    if threshold_range.numel() == 0:
        threshold = valid_pixels.max()
    else:
        threshold_scores = otsu_intraclass_variance(valid_pixels, threshold_range)
        threshold = threshold_range[torch.argmin(threshold_scores)]
    return threshold

class OptimizedDetectionPredictor(DetectionPredictor):
    def __init__(self, *args, regionprops=(), **kwds):
        super().__init__(*args, **kwds)
        self.regionprops = regionprops
        self._last_preprocessed = None
    
    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        # Slightly optimized for grayscale images of fixed size.
        im = np.stack(im)
        im = torch.from_numpy(im[..., 0]).to(self.model.device)
        im = im[:, None].expand(-1, 3, -1, -1) # (B, 3, H, W) — zero-copy view on GPU
        if getattr(self.args, "half", False):
            im = im.half() / 255
        else:        
            im = im.float() / 255
        self._last_preprocessed = im
        return im


def export_tensorrt_engine_with_progress(parent, model, roi_size, half, batch_size):
    """Export a YOLO model to TensorRT while keeping the UI responsive."""
    progress = QtWidgets.QDialog(parent)
    progress.setWindowTitle("Preparing TensorRT model")
    progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    progress.setModal(True)
    progress.setMinimumWidth(380)
    layout = QtWidgets.QVBoxLayout(progress)
    layout.addWidget(
        QtWidgets.QLabel("Exporting TensorRT engine. This can take a while...")
    )
    bar = QtWidgets.QProgressBar()
    bar.setRange(0, 0)
    layout.addWidget(bar)

    result = {}
    error = {}

    def _do_export():
        try:
            result["exported_file"] = model.export(
                format="engine",
                imgsz=roi_size,
                nms=True,
                batch=batch_size,
            )
        except Exception as ex:
            error["exception"] = ex

    worker = threading.Thread(target=_do_export, name="TensorRTExport", daemon=True)
    worker.start()
    progress.show()
    while worker.is_alive():
        QtWidgets.QApplication.processEvents()
        time.sleep(0.05)
    progress.close()

    if "exception" in error:
        raise error["exception"]
    return result["exported_file"]

class YoloBackgroundRemover(QtCore.QThread):
    frame_processed = QtCore.Signal(int, str, np.ndarray)
    preview_image = QtCore.Signal(str, np.ndarray, np.ndarray)

    def __init__(
        self,
        progress_dialog,
        task_queue,
        bg_params,
        file_write_params,
        inference_params,
        video_queue,
        file_queue,
        link_tracks,
        track_settings,        
    ):
        super().__init__(parent=progress_dialog)
        self.progress_dialog = progress_dialog
        self.task_queue = task_queue
        self.bg_params = bg_params
        self.file_write_params = file_write_params        
        self.inference_params = dict(inference_params)
        self.link_tracks = link_tracks
        self.track_settings = track_settings
        self.video_tasks = video_queue
        self.file_tasks = file_queue
        self._last_idx = -1
        # We store the original file names for later
        self._orig_fnames = {}
        self.buffer = []
        model = YOLO(self.inference_params["model_file"], task="detect")

        if inference_params["tensorRT"] and not inference_params["model_file"].endswith(".engine"):
            roi_slice = self.bg_params["roi_slice"]
            roi_size = (
                roi_slice[0].stop - roi_slice[0].start,
                roi_slice[1].stop - roi_slice[1].start,
            )
            exported_file = export_tensorrt_engine_with_progress(
                progress_dialog,
                model,
                roi_size,
                self.inference_params["half_precision"],
                self.inference_params["batch_size"],
            )
            self.model = YOLO(exported_file, task="detect")
        else:
            self.model = model
        
        self.track_file_queue = QueueWithSignals(
            queue.Queue(), "Track file writing", parent=self
        )
        self.track_file_writer = TrackFileThread(
            progress_dialog,
            self.file_write_params["target_folder"],
            self.track_file_queue,
            link_tracks=self.link_tracks,
            track_settings=self.track_settings,
            fps=self.file_write_params["fps"],
            pixel_size=self.track_settings["pixel_size"],
        )

    def start(self, *args, **kwds):
        super().start(*args, **kwds)
        self.track_file_writer.start()

    def find_cells(self, frames, conf=0.2, iou=0.7, half=False):
        with torch.no_grad():
            if self.link_tracks and self.track_settings["package"] == "yolo":
                tracker_config = os.path.join(
                    "config", self.track_settings["yolo"]["tracker_type"] + ".yaml"
                )
                results = self.model.track(
                    frames,
                    imgsz=frames[0].shape[:2],
                    batch=self.inference_params["batch_size"],
                    rect=False,
                    conf=conf,
                    iou=iou,
                    tracker=tracker_config,
                    persist=True,
                    predictor=OptimizedDetectionPredictor,
                    end2end=False,
                    device=DEVICE,
                )
            else:
                results = self.model.predict(
                    frames,
                    imgsz=frames[0].shape[:2],
                    batch=self.inference_params["batch_size"],
                    rect=False,
                    conf=conf,
                    iou=iou,
                    predictor=OptimizedDetectionPredictor,
                    end2end=False,
                    device=DEVICE,
                )
        # We only use one of the channels – they are all the same
        gray_batch = self.model.predictor._last_preprocessed[:, 0]
        boxes_batch = [
            result.boxes.xyxy.to(device=self.model.predictor._last_preprocessed.device)
            for result in results
        ]
        features = extract_patches_centroid_theta(gray_batch, boxes_batch)                
        augmented_results = []
        for r, f in zip(results, features):
            result_set = {
                "boxes": f["boxes_int"].cpu(),
                "masked_image": f["masked_image"].cpu().numpy(),
                "orientation": f["orientation"],
                "centroid": f["centroid"],
                "major_axis_length": f["major_axis_length"],
                "minor_axis_length": f["minor_axis_length"],
                "conf": r.boxes.conf,
            }
            if r.boxes.is_track:
                result_set["id"] = r.boxes.id.int().cpu().numpy()
            augmented_results.append(result_set)

        torch.cuda.empty_cache()
        return augmented_results

    def handle_frame(self, masked_image, frame, epoch, relative_idx, idx):        
        # We start our file names with 1 for ffmpeg
        if epoch == -1:
            filename = os.path.join(
                "frames", self.bg_params["filename_prefix"] + f"{idx + 1:07d}.tiff"
            )
        else:
            filename = os.path.join(
                "frames",
                self.bg_params["filename_prefix"] + f"{epoch:04d}_{relative_idx + 1:07d}.tiff",
            )
        logger.debug(
            f"Background remover: processed frame {idx} ('{filename}')",
            extra={"index": idx},
        )
        self.file_tasks.put(
            {
                "type": "frame",
                "idx": idx,
                "fname": filename,
                "frame": masked_image,
            }
        )        

    def handle_buffer(self, idx, epoch, relative_idx):
        buffer_size = len(self.buffer)
        if buffer_size == 0:
            return  # nothing to do

        try:
            logger.debug(
                    f"Finding cells in frames {idx-buffer_size}–{idx}",
                    extra={"index": idx},
                )
            results = self.find_cells(
                self.buffer,
                conf=self.inference_params["conf_threshold"],
                iou=self.inference_params["iou"],
                half=self.inference_params["half_precision"],
            )
            bounding_boxes = [r["boxes"] for r in results]
            masked_images = [r["masked_image"] for r in results]
            orientations = [r["orientation"] for r in results]
            major_axis_length = [r["major_axis_length"] for r in results]
            minor_axis_length = [r["minor_axis_length"] for r in results]
            centroids = [r["centroid"] for r in results]
            conf = [r["conf"] for r in results]
            if self.link_tracks and self.track_settings["package"] == "yolo":
                track_ids = []
                for i, r in enumerate(results):
                    if (track_id := r.get("id", None)) is not None:
                        track_ids.append(track_id)
                    else:
                        logger.warning(
                            f"Yolo did not return IDs for frame {idx - buffer_size + i + 1}"
                        )
                        track_ids.append(-1*torch.ones_like(r["orientation"]))

            # Write results to track file
            track_task = {
                "idx": idx,
                "epoch": epoch,
                "relative_start_idx": relative_idx - buffer_size + 1,
                "n_frames": buffer_size,
                "bounding_boxes": bounding_boxes,
                "orientations": orientations,
                "major_axis_length": major_axis_length,
                "minor_axis_length": minor_axis_length,
                "centroids": centroids,
                "conf": conf,
            }
            if self.link_tracks and self.track_settings["package"] == "yolo":
                track_task["track_ids"] = track_ids

            self.track_file_queue.put(track_task)

            for i, (orig_frame, masked_image) in enumerate(
                zip(self.buffer, masked_images)
            ):
                self.handle_frame(
                    masked_image,
                    orig_frame,
                    epoch,
                    relative_idx - buffer_size + i + 1,
                    idx - buffer_size + i + 1,
                )
                self.task_queue.task_done(index=idx - buffer_size + i + 1, measure=False)
            self.buffer.clear()

            if self.video_tasks is not None:
                logger.debug(
                    f"Background remover: submitted video task for frames {idx - buffer_size + 1}–{idx}",
                    extra={"index": idx},
                )
                self.video_tasks.put(
                    {
                        "idx": idx,
                        "n_frames": buffer_size,
                        "epoch": epoch,
                        "relative_start_idx": relative_idx - buffer_size + 1,
                        "discard": False,
                    }
                )

        except Exception:
            logger.exception(
                "Error while extracting cells in frame", extra={"index": idx}
            )
            raise
        finally:
            self.buffer.clear()

    def finish_frames(self, epoch, relative_idx_start, final, discard):
        if not discard:
            self.handle_buffer(self._last_idx, epoch, relative_idx_start)

        if self.video_tasks is not None:
            # Request merging of video files
            self.video_tasks.put(
                {"stop": True, "final": final, "epoch": epoch, "discard": discard}
            )

        self.track_file_queue.put({"stop": True, "final": final, "epoch": epoch})

    @run_wrapper
    def run(self) -> None:
        global exception_occured
        while True:
            idx, task = self.task_queue.get()  # blocks until task is available
            logger.debug(
                f"Received new task with idx {idx} '({task})'", extra={"index": idx}
            )

            assert idx > self._last_idx, (idx, self._last_idx)

            if idx > self._last_idx + 1 and not (
                task["type"] == "stop" and task["final"]
            ):  # stop signals use sys.maxsize as index
                # The first file in queue is not the one that should be treated next
                logger.debug(
                    f"First file in queue is {idx}, but didn't treat {self._last_idx + 1} yet - will wait",
                    extra={"index": idx},
                )
                # Put task back in queue
                self.task_queue.put_back((idx, task))
                self.thread().msleep(100)
                continue            

            if task["type"] == "stop":
                self.finish_frames(
                        task["epoch"],
                        task["epoch_start_idx"],
                        task["final"],
                        task["discard"],
                    )
                self.task_queue.task_done(index=idx, measure=False)
                self._last_idx = idx

                if task.get("final", False):
                    # End thread
                    break
                else:
                    continue

            self._last_idx = idx
            fname = task["fname"]
            discard = task.get("discard", False)

            if discard:
                logger.debug(f"Discarding file {idx} ('{fname}')", extra={"index": idx})
                self.task_queue.task_done(index=idx, measure=False)
                continue

            frame = task["frame"]
            epoch = task["epoch"]
            relative_idx = task["relative_idx"]
            # The ROI has been "frozen", we only care about this part of the frame from now on
            frame = frame[self.bg_params["roi_slice"]]
            if self.bg_params["dark_field"]:
                frame = 255 - frame
            self.buffer.append(frame)

            buffer_size = self.inference_params["batch_size"]
            if len(self.buffer) == buffer_size:
                self.handle_buffer(idx, epoch, relative_idx)

        # Block until tracking thread finishes
        self.track_file_writer.wait()

class FileWriterThread(QtCore.QThread):
    def __init__(
        self,
        parent,
        source_folder,
        target_folder,
        compression_algorithm,
        fps,
        task_queue,
        track_queue=None,
        wait_for=1,
        delete_files=False,
        delete_compressed_files=False,
    ):
        super().__init__(parent=parent)
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.compression_algorithm = compression_algorithm
        self.fps = fps
        self.delete_files = delete_files
        self.delete_compressed_files = delete_compressed_files
        self.task_queue = task_queue
        self.track_queue = track_queue
        self._stop_received = 0
        self._stopped = False
        self.wait_for = wait_for

    @run_wrapper
    def run(self):
        while not (self._stopped and self.task_queue.empty()):
            try:
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue  # maybe stop was requested?
            if task["type"] == "frame":
                idx, fname, frame = task["idx"], task["fname"], task["frame"]
                self.receive_frame(idx, fname, frame)
            elif task["type"] == "delete":
                source_path, idx = task["fname"], task["idx"]
                self.delete_file(source_path, idx)
            else:
                raise AssertionError("Expected 'frame', or 'delete' as type")

    def delete_file(self, fname, idx):
        logger.debug(
            f"Received request to delete file {idx} ('{fname}')", extra={"index": idx}
        )
        # delete original file
        try:
            os.remove(fname)
            logger.debug(
                f"Deleted file '{os.path.basename(fname)}", extra={"index": idx}
            )
            self.task_queue.task_done(idx, measure=False)
        except Exception as ex:
            logger.error(
                f"Could not delete file '{os.path.basename(fname)}: {str(ex)} (will try again later",
                extra={"index": idx},
            )
            # Put task back into queue
            self.task_queue.put_back({"type": "delete", "fname": fname, "idx": idx})

    @QtCore.Slot(int, str, np.ndarray)
    def receive_frame(self, idx, fname, array):
        logger.debug(
            f"Received frame {idx} for writing to '{fname}'", extra={"index": idx}
        )
        full_path = os.path.join(self.target_folder, fname)
        dir_name = os.path.dirname(
            full_path
        )  # might be at a deeper level than the target folder, e.g. for background
        os.makedirs(dir_name, exist_ok=True)
        # libtiff seems to be slower for writing than imageio…
        write_image(
            full_path,
            array,
            compression=self.compression_algorithm,
        )
        logger.debug(f"Wrote '{fname}' (with imageio)", extra={"index": idx})
        self.task_queue.task_done(index=idx)

    def stop(self):
        self._stop_received += 1
        if self._stop_received >= self.wait_for:
            logger.info(f"Stopping file writer thread ({self.objectName()})")
            self._stopped = True
        else:
            logger.info(
                f"Received stop but still waiting for {self.wait_for-self._stop_received} more signals ({self.objectName()})"
            )


class MulitiplierSpinBox(QtWidgets.QSpinBox):
    def __init__(self, main_window, n_values):
        super().__init__()
        self.main_window = main_window
        self.setMinimum(0)
        self.setMaximum(n_values)
        self.setSpecialValueText("Never")
        self.setValue(0)

    def textFromValue(self, val: int) -> str:
        if value := self.main_window.background_frames.value():
            return str(value * val)
        else:
            return ""


class DivisorSpinBox(QtWidgets.QSpinBox):
    def __init__(self, main_window, max_value):
        super().__init__()
        self.main_window = main_window
        self.divisors = None
        self.update_divisors(max_value)

    def update_divisors(self, max_value):
        current_value = self.divisors[self.value() - 1] if self.divisors else -1
        self.divisors = divisors(max_value)
        self.setMinimum(1)
        self.setMaximum(len(self.divisors))
        if current_value == -1:
            self.setValue(len(self.divisors))
        else:
            # find the closest value
            closest_idx = np.argmin(np.abs(np.array(self.divisors) - current_value))
            self.setValue(closest_idx + 1)

    def textFromValue(self, val: int) -> str:
        if len(self.divisors) == 0:
            return ""
        return str(self.divisors[val - 1])


class InitialBackgroundCalculator:
    """
    Class to calculate the initial background from a set of frames. This class does not use the GPU,
    and stores all the frames in memory – this way, we can still change the ROI and the number of
    frames used for calculating the background.
    """

    def __init__(self, width, height, dtype=np.uint8):
        self.width = width
        self.height = height
        self.dtype = dtype
        self.frames = np.empty((0, height, width), dtype=dtype)
        self._n_frames = 0
        # We cache the background to not recalculate e.g. for switch to dark field
        self._prev_background = None
        self._prev_bg_frames = None

    def resize(self, n_frames):
        if n_frames < self.frames.shape[0]:
            return  # nothing to do
        self.frames.resize((n_frames, self.height, self.width), refcheck=False)
        self._n_frames = min(n_frames, self._n_frames)

    def add_frame(self, frame):
        self.frames[self._n_frames] = frame
        self._n_frames += 1

    def calc_background(self, n_frames):
        assert n_frames <= self._n_frames, (n_frames, self.n_frames)

        if self._prev_background is not None and self._prev_bg_frames == n_frames:
            return self._prev_background

        # Calculate on GPU
        frames = self.frames[:n_frames]
        bg = np.mean(frames, axis=0).astype(self.dtype)
        self._prev_background = bg
        self._prev_bg_frames = n_frames
        return bg

    def remove_background(self, frame, background, threshold):
        background_inv = np.clip(
            np.invert(background).astype("int16") + threshold, 0, 255
        ).astype("uint8")
        removed = np.invert(
            np.clip(np.invert(frame), background_inv, None) - background_inv
        )
        return removed


DEFAULT_BUFFERSIZE = 50


class VideoThread(QtCore.QThread):
    frame_processed = QtCore.Signal(int, str)

    def __init__(
        self,
        parent,
        movie_dir,
        fname_prefix,
        task_queue,
        file_queue,
        delete_compressed=False,
        fps=15,
    ):
        super().__init__(parent)
        self.movie_dir = movie_dir
        self.image_dir = os.path.dirname(fname_prefix)
        self.fname_prefix = fname_prefix
        self.task_queue = task_queue
        self.file_queue = file_queue
        self.delete_compressed = delete_compressed
        self.fps = fps
        self.processes_and_dirs = []
        self.movie_list = []

    def check_processes(self, block=False):
        global exception_occured
        logger.debug("Checking whether any running processes have finished")
        for process, temp_movie_dir, frames, fnames, idx in list(
            self.processes_and_dirs
        ):  # copy the list to be able to remove items
            if block:
                error_code = process.wait()
            else:
                error_code = process.poll()
            if error_code is not None:
                if error_code:
                    logger.error(
                        f"Error in video generation process {temp_movie_dir} (error code {error_code})"
                    )
                    exception_occured = True
                else:
                    logger.debug(f"Process {temp_movie_dir} has finished")
                    for frame_idx in frames:  # signal remaining frames
                        frame_fname = os.path.join(
                            self.image_dir, "frames", fnames[frame_idx]
                        )
                        self.frame_processed.emit(frame_idx, frame_fname)
                        if self.delete_compressed:
                            self.file_queue.put(
                                {
                                    "type": "delete",
                                    "idx": frame_idx,
                                    "fname": frame_fname,
                                }
                            )
                    shutil.rmtree(temp_movie_dir)
                self.task_queue.task_done(index=idx)
                self.processes_and_dirs.remove(
                    (process, temp_movie_dir, frames, fnames, idx)
                )

    def run(self):
        global exception_occured
        threading.current_thread().name = QtCore.QThread.currentThread().objectName()
        if TRACE:
            get_tracer().enable_thread_tracing()
        os.makedirs(self.movie_dir, exist_ok=True)
        ffmpeg_binary = ffmpeg_plugin.get_exe()
        # We call ffmpeg in a subprocess, and parse its results from the stdout pipe
        fps = str(self.fps)
        
        while True:
            self.check_processes()
            try:
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue
            logger.debug("New task for VideoThread: " + str(task))

            if task.get('stop', False):
                logger.info("Received stop signal, merging videos")
                # Make sure that all processes have finished
                self.check_processes(block=True)

                if not task[
                    "discard"
                ]:  # This might be called at the end during a non-recording epoch
                    self.join_videos(self.movie_list, ffmpeg_binary, task["epoch"])

                self.task_queue.task_done()

                if task["final"]:
                    logger.info("Stopping VideoThread")
                    break
                else:
                    self.movie_list.clear()
                    continue

            last_idx, n_frames, epoch, relative_start_idx = (
                task["idx"],
                task["n_frames"],
                task["epoch"],
                task["relative_start_idx"],
            )
            if epoch == -1:
                start_idx = last_idx - n_frames + 1
            else:
                start_idx = relative_start_idx - n_frames + 1
            frames = np.arange(start_idx, start_idx + n_frames)
            # Unfortunately, ffmpeg only accepts either globs or sequences of files, but does not have
            # a way to select a specific set of files. To work around this limitation, we create hardlinks
            # in a separate directory and delete them afterwards
            temp_movie_dir = os.path.join(
                self.movie_dir, f"temp{last_idx-n_frames+1}_{last_idx+1}"
            )
            if epoch == -1:
                fnames = {
                    frame: f"{os.path.basename(self.fname_prefix)}{frame+1:07d}.tiff"
                    for frame in frames
                }
                input_fname_pattern = os.path.abspath(
                    os.path.join(
                        temp_movie_dir,
                        os.path.basename(self.fname_prefix) + "%07d.tiff",
                    )
                )
            else:
                fnames = {
                    frame: f"{os.path.basename(self.fname_prefix)}{epoch:04d}_{frame+1:07d}.tiff"
                    for frame in frames
                }
                input_fname_pattern = os.path.abspath(
                    os.path.join(
                        temp_movie_dir,
                        os.path.basename(self.fname_prefix) + f"{epoch:04d}_%07d.tiff",
                    )
                )
            os.mkdir(temp_movie_dir)
            for fname in fnames.values():
                src, dest = os.path.join(self.image_dir, "frames", fname), os.path.join(
                    temp_movie_dir, fname
                )
                while not verify_tiff(src):
                    logger.debug(f"Waiting for file '{src}' to be written")
                    self.thread().msleep(100)
                logger.debug(f"File '{src}' looks fine, going ahead with video")
                os.link(src, dest)
            movie_fname = os.path.abspath(
                os.path.join(self.movie_dir, f"part_{len(self.movie_list):06d}.mp4")
            )
            self.movie_list.append(movie_fname)

            ffmpeg_call = [
                ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                fps,
                "-start_number",
                str(start_idx),
                "-i",
                input_fname_pattern,
                "-an",
                "-y",
                "-c:v",
                FFMPEG_VCODEC,
                "-preset",
                FFMPEG_PRESET,
                "-pix_fmt",
                FFMPEG_PIX_FMT,
                "-r",
                fps,
                movie_fname,
            ]
            logger.info(
                "Generating video part using ffmpeg: "
                + (" ".join(ffmpeg_call)).replace("%", "%%")
            )
            # Wait if there are two many processes running in parallel
            while len(self.processes_and_dirs) >= MAX_FFMPEG_PROCESSES:
                logger.debug(
                    f"Too many ffmpeg processes running ({len(self.processes_and_dirs)}), waiting for one to finish"
                )
                self.check_processes()
                self.thread().msleep(500)
            process = subprocess.Popen(ffmpeg_call)
            self.processes_and_dirs.append(
                (process, temp_movie_dir, frames, fnames, last_idx)
            )

        logger.info("VideoThread finished")

    def join_videos(self, movie_list, ffmpeg_binary, epoch):
        global exception_occured
        if not movie_list:
            logger.info("No movies to join")
            return

        # All movies have been written, join them
        # Try to find a useful file name
        logger.info(f"Joining movies for epoch {epoch}")
        file_name = filename_prefix([os.path.basename(self.fname_prefix)], prefix=False)
        if not file_name:
            file_name = "video"
        if epoch != -1:
            file_name += f"_{epoch:04d}"
        file_name += ".mp4"
        file_list = os.path.abspath(
            os.path.join(self.movie_dir, file_name + "-file_list.txt")
        )
        logger.debug(f"Joined movie filename: {file_name}, file list: {file_list}")
        with open(file_list, "w") as f:
            for movie_fname in movie_list:
                f.write(f"file '{movie_fname}'\n")
        ffmpeg_call = [
            ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            file_list,
            "-c",
            "copy",
            os.path.abspath(os.path.join(self.movie_dir, file_name)),
        ]
        logger.info("joining movies via ffmpeg:" + (" ".join(ffmpeg_call)))
        try:
            subprocess.run(ffmpeg_call, check=True)
            # Only delete the parts if the join was successful
            for movie_fname in movie_list:
                os.remove(movie_fname)
            os.remove(file_list)
        except subprocess.CalledProcessError:
            logger.exception("Could not join movies")
            exception_occured = True


class TrackFileThread(QtCore.QThread):
    def __init__(
        self,
        parent,
        target_dir,
        track_queue,
        link_tracks,
        track_settings,
        fps,
        pixel_size
    ):
        super().__init__(parent)
        self.tracking_dir = os.path.join(target_dir, "tracking")
        os.makedirs(self.tracking_dir, exist_ok=False)
        self.track_file_list = []
        self.track_queue = track_queue
        self.link = link_tracks
        self.track_settings = track_settings
        self.fps = fps
        self.pixel_size = pixel_size

    def run(self):
        global exception_occured
        threading.current_thread().name = QtCore.QThread.currentThread().objectName()
        if TRACE:
            get_tracer().enable_thread_tracing()

        while True:            
            try:
                task = self.track_queue.get(timeout=1)
            except queue.Empty:
                continue
            logger.debug("New task for TrackFileThread: " + str(task))

            if task.get('stop', False):
                logger.info("Received stop signal, merging tracks")

                self.join_tracks(task.get("epoch", -1))

                self.track_queue.task_done(measure=False)

                if task["final"]:
                    logger.info("Stopping TrackFileThread")
                    break
                else:
                    self.track_file_list.clear()
                    continue

            (
                last_idx,
                n_frames,
                epoch,
                relative_start_idx,
                bounding_boxes,
                orientations,
                major_lengths,
                minor_lengths,
                centroids,
                confs,
            ) = (
                task["idx"],
                task["n_frames"],
                task["epoch"],
                task["relative_start_idx"],
                task["bounding_boxes"],
                task["orientations"],
                task["major_axis_length"],
                task["minor_axis_length"],
                task["centroids"],
                task["conf"],
            )
            track_ids = task.get("track_ids", None)
            if epoch == -1:
                start_idx = last_idx - n_frames + 1
            else:
                start_idx = relative_start_idx - n_frames + 1

            if epoch == -1:
                fname = os.path.join(self.tracking_dir, f"tracking_unlinked_{self.fps:.01f}_fps_{self.pixel_size:.02f}_um_{start_idx:07d}-{last_idx:07d}.tsv")
            else:
                fname = os.path.join(self.tracking_dir, f"tracking_unlinked_{self.fps:.01f}_fps_{self.pixel_size:.02f}_um_{epoch:04d}_{start_idx:07d}-{last_idx:07d}.tsv")

            self.track_file_list.append(fname)

            # We write the file manually, no need to go through pandas
            with open(fname, "wt") as f:
                if self.link and self.track_settings["package"] == "yolo":
                    for frame, (track_id, center, boxes, angles, majors, minors, conf) in enumerate(
                        zip(track_ids, centroids, bounding_boxes, orientations, major_lengths, minor_lengths, confs)
                    ):
                        # No headers for easier merging
                        for track, (x, y), (b0, b1, b2, b3), angle, major, minor, c in zip(
                            track_id, center, boxes, angles, majors, minors, conf
                        ):
                            if track >= 0:
                                f.write(
                                    f"{frame + start_idx}\t{int(track)}\t{x}\t{y}\t{b0}\t{b1}\t{b2}\t{b3}\t{angle:.2f}\t{major:.2f}\t{minor:.2f}\t{c:.2f}\n"
                                )
                            else:
                                # Do not write any idea if Yolo did not return one
                                f.write(
                                    f"{frame + start_idx}\t\t{x}\t{y}\t{b0}\t{b1}\t{b2}\t{b3}\t{angle:.2f}\t{major:.2f}\t{minor:.2f}\t{c:.2f}\n"
                                )
                            
                else:
                    for frame, (center, boxes, angles, majors, minors, conf) in enumerate(
                        zip(centroids, bounding_boxes, orientations, major_lengths, minor_lengths, confs)
                    ):
                        # No headers for easier merging
                        for (x, y), (b0, b1, b2, b3), angle, major, minor, c  in zip(
                            center, boxes, angles, majors, minors, conf
                        ):
                            f.write(
                                f"{frame + start_idx}\t{x}\t{y}\t{b0}\t{b1}\t{b2}\t{b3}\t{angle:.2f}\t{major:.2f}\t{minor:.2f}\t{c:.2f}\n"
                            )
            self.track_queue.task_done(last_idx)

        logger.info("TrackFileThread finished")

    def join_tracks(self, epoch):
        logger.debug("Joining track files")
        if epoch == -1:
            fname = os.path.join(self.tracking_dir, f"tracking_unlinked_{self.fps:.01f}_fps_{self.pixel_size:.02f}_um.tsv")
        else:
            fname = os.path.join(self.tracking_dir, f"tracking_unlinked_{epoch:04d}_{self.fps:.01f}_fps_{self.pixel_size:.02f}_um.tsv")
        
        # Concatenate files
        with open(fname, "wt") as out_f:
            # Write header
            if self.link_tracks and self.track_settings["package"] == "yolo":
                out_f.write("frame\tid\tx\ty\tbbox-0\tbbox-1\tbbox-2\tbbox-3\tangle\tlength\twidth\tconf\n")
            else:
                out_f.write("frame\tx\ty\tbbox-0\tbbox-1\tbbox-2\tbbox-3\tangle\tlength\twidth\tconf\n")
            for in_fname in self.track_file_list:
                with open(in_fname, "rt") as in_f:
                    shutil.copyfileobj(in_f, out_f)
        # Delete individual files
        for in_fname in self.track_file_list:
            os.remove(in_fname)

        if self.link:
            self.link_tracks(epoch, fname)


    def link_tracks(self, epoch, tracking_fname):
        track_folder = os.path.dirname(tracking_fname)
        cells_df = pd.read_csv(tracking_fname, sep="\t")
        try:
            logger.info(f"Linking tracks for epoch {epoch}")
            linked = self.link_wrapper(cells_df)                
            if self.track_settings["movement_features"]:
                segments = segments_from_table(linked)
                logger.debug(f"Calculating features over {len(segments)} segments")
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    segments = executor.map(calculate_features, segments)
                    segments = executor.map(mark_avoiding_reactions_from_motion, segments)
                linked = pd.concat(segments)
                linked.sort_values(by='frame')
                logger.debug("Finished calculating features")
                if epoch != -1:
                    linked_fname = os.path.join(track_folder, f"tracking_{epoch:07d}_linked_with_features_{self.fps:.1f}_fps_1_um.tsv")
                else:
                    linked_fname = os.path.join(track_folder, f"tracking_linked_with_features_{self.fps:.1f}_fps_1_um.tsv")
            else:
                if epoch != -1:
                    linked_fname = os.path.join(track_folder, f"tracking_{epoch:07d}_linked_{self.fps:.1f}_fps_1_um.tsv")
                else:
                    linked_fname = os.path.join(track_folder, f"tracking_linked_{self.fps:.1f}_fps_1_um.tsv")
            # ↑ Note that the filename states 1_um, since linked tracks are already scaled to µm
            if self.track_settings["zip_tracking_file"]:
                linked_fname += ".gz"
            linked.to_csv(linked_fname, sep="\t", index=False, float_format="%.2f")
            logger.info(f"linking tracks for epoch {epoch} done, linked tracks saved to {linked_fname}")
        except Exception:
            logger.exception(f"Linking tracks for epoch {epoch} failed")

    def link_wrapper(self, df):
        package = self.track_settings["package"]
        settings = dict(self.track_settings[package])
        # Preparations common to all packages
        # 1. Scale the coordinates and lengths to µm
        df[["x", "y", "bbox-0", "bbox-1", "bbox-2", "bbox-3"]] *= self.pixel_size
        if "length" in df.columns:
            df[["length", "width"]] *= self.pixel_size

        # Yolo did the tracking on-line, so nothing else is left to do
        if package == "yolo":
            return df
        
        # 2. Convert search range and memory to µm and frames
        search_range = settings["maximum_speed"] / self.fps
        memory = int(round(settings["memory"] * self.fps))
        del settings["maximum_speed"]
        del settings["memory"]
        if package == "trackpy":
            import trackpy as tp
            if settings["adaptive_stop"] == 0:
                adaptive_stop = None
            else:
                adaptive_stop = settings["adaptive_stop"] / self.fps
            del settings["adaptive_stop"]
            logger.info(
                f"Linking tracks with Trackpy, search_range={search_range}, memory={memory}, adaptive_stop={adaptive_stop}, {' '.join(f'{k}={v}' for k, v in settings.items())}"
            )
            result = tp.link(df, search_range=search_range, memory=memory, adaptive_stop=adaptive_stop, **settings)
            result.rename(columns={"particle": "id"}, inplace=True)

        elif package == "norfair":
            import norfair
            initialization_delay = int(round(settings["initialization_delay"] * self.fps))
            del settings["initialization_delay"]
            logger.info(f"Linking tracks with Norfair, distance_threshold={search_range}, hit_counter_max={memory}, initialization_delay={initialization_delay}, {' '.join(f'{k}={v}' for k, v in settings.items())}")
            tracker = norfair.Tracker(
                distance_function="mean_euclidean",
                distance_threshold=search_range,
                initialization_delay=initialization_delay,
                hit_counter_max=memory,
            )
            output = []
            for frame, rows in df.groupby('frame'):
                norfair_detections = [norfair.Detection(points=np.array([row['x'], row['y']]), data=row) for _, row in rows.iterrows()]
                tracked_objects = tracker.update(detections=norfair_detections)
                for object in tracked_objects:
                    last_detection = object.last_detection.data
                    if last_detection['frame'] == frame: ## the last detection could be far in the past
                        row = last_detection.to_dict()
                        row.update({'id' : object.id})
                        output.append(row)
            result = pd.DataFrame(output)
        else:
            raise ValueError(f"Unknown tracking package '{package}'")
        return result

def log_namer(name):
    return name + ".gz"


def log_rotator(source, dest):
    with open(source, "rb") as f_in:
        with gzip.open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(source)


class QueueWithSignals(QtCore.QObject):
    """
    Wrapper class around a Python queue (e.g. queue.Queue or queue.PriorityQueue) that emits signals when tasks are added or done.
    """

    task_added = QtCore.Signal(int)  # Total tasks in queue
    task_finished = QtCore.Signal(int, float)  # Total tasks in queue, time taken

    def __init__(self, queue, name, parent=None):
        super().__init__(parent=parent)
        self.counter = 0
        self.lock = threading.Lock()
        self._queue = queue
        self._name = name
        self._last_get = {}

    def get(self, *args, **kwds):
        task = self._queue.get(*args, **kwds)
        index = self._extract_task_index(task)
        self._last_get[index] = time.time()
        return task

    def _extract_task_index(self, task):
        if isinstance(task, tuple):
            index = task[0]
        else:
            index = task.get("idx", task.get("index", None))
        return index

    def empty(self):
        return self.counter == 0

    def qsize(self):
        return self.counter

    def put(self, *args, **kwds):
        self.lock.acquire()
        self.counter += 1
        self.lock.release()
        try:
            logger.debug(
                f"{self._name}: Adding task to queue with size {self._queue.qsize()}, counter: {self.counter}"
            )
        except NotImplementedError:  # on macOS
            logger.debug(
                f"{self._name}: Adding task to queue, counter: {self.counter}"
            )
        self._queue.put(*args, **kwds)
        self.task_added.emit(self.counter)

    def put_back(self, task):
        logger.debug(f"{self._name}: Task put back for later processing")
        # Don't measure time or signal tasks for tasks that are put back
        self._queue.task_done()
        self._queue.put(task)

    def task_done(self, index=None, measure=True):
        self.lock.acquire()
        self.counter -= 1
        self.lock.release()
        if measure:
            took = time.time() - self._last_get[index]
            logger.debug(f"{self._name}: Task {index} finished in {took:.3f} s")
        else:
            took = float("nan")  # cannot be None, since we send a float
            logger.debug(f"{self._name}: Task {index} finished (not measuring)")
        self._queue.task_done()

        self.task_finished.emit(self.counter, took)


RATE_COLUMN_WIDTH = 100


class WaitThread(QtCore.QThread):
    def __init__(self, parent, threads):
        super().__init__(parent=parent)
        self.threads = threads

    def run(self):
        threading.current_thread().name = QtCore.QThread.currentThread().objectName()
        if TRACE:
            get_tracer().enable_thread_tracing()

        # Wait for all given threads to finish
        for thread in self.threads:
            thread.wait()

        logger.info("WaitThread finished")

class ProgressDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent,
        bg_params,
        file_write_params,
        fileno_offset,
        fileno_step,      
        inference_params,
        record_video,
        link_tracks,
        track_settings,
        read_function,
        schedule,
        archive_compressed_files,
    ):
        super().__init__(parent=parent)
        # Zip the log files if they become too big
        log_fname = os.path.join(
            file_write_params["target_folder"],
            f"log_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.log",
        )
        os.makedirs(os.path.dirname(log_fname), exist_ok=True)
        rh = RotatingFileHandler(log_fname, maxBytes=50_000_000, backupCount=10_000)
        rh.rotator = log_rotator
        rh.namer = log_namer
        self.log_file = rh
        self.log_file.setLevel(FILE_LOG_LEVEL)
        self.log_file.setFormatter(
            logging.Formatter(
                '%(created)s\t%(threadName)s\t%(levelname)s\t%(funcName)s:%(lineno)s\t%(index)d\t"%(message)s"'
            )
        )
        logger.addHandler(self.log_file)
        logger.addFilter(default_index_filter)
        logger.info("Starting")
        self.setWindowTitle("Background removal progress")
        self.available_files = 0
        self._last_available_files = 0
        self._available_files_per_s = 0
        self.discarded_files = 0
        self.counters = {
            "background_removal": 0,
            "files_read": 0,
            "files_written": 0,
            "videos_written": 0,
            "tracking": 0,
        }
        self.buffer_sizes = self.counters.copy()
        self.counters_per_s = self.counters.copy()
        self.written_files_fnames = []
        self.last_ctime = None
        self.estimated_framerate = 0
        self.inference_params = inference_params
        self.record_video = record_video
        self.link_tracks = link_tracks
        self.track_settings = track_settings
        self.bg_params = bg_params
        self.file_write_params = file_write_params
        self.fileno_offset = fileno_offset
        self.fileno_step = fileno_step
        self.read_function = read_function
        self.archive_compressed_files = archive_compressed_files

        # Left: progress reports
        progress = QtWidgets.QGroupBox("Progress")

        progress_layout = QtWidgets.QGridLayout()
        # Source files
        progress_layout.addWidget(QtWidgets.QLabel("Files"), 0, 0, 1, 1)
        self.source_file_total = QtWidgets.QLabel()
        progress_layout.addWidget(self.source_file_total, 0, 1, 1, 1)
        self.source_file_rate = QtWidgets.QLabel()
        self.source_file_rate.setAlignment(Qt.AlignRight)
        progress_layout.addWidget(self.source_file_rate, 0, 2, 1, 1)

        self.file_read_queue = QueueWithSignals(
            queue.PriorityQueue(), "File read", parent=self
        )
        self.file_write_queue = QueueWithSignals(
            queue.Queue(), "File write", parent=self
        )
        self.processing_queue = QueueWithSignals(
            queue.PriorityQueue(), "Background removal", parent=self
        )
        self.video_queue = None
        self.queues = {
            "Files read": self.file_read_queue,
            "Background removal": self.processing_queue,
            "Files written": self.file_write_queue,
        }
        if record_video:
            self.video_queue = QueueWithSignals(
                queue.Queue(), "Video writing", parent=self
            )
            self.queues["Videos written"] = self.video_queue        

        self.progress_bars = {}

        if schedule:
            progress_layout.addWidget(QtWidgets.QLabel("Schedule"), 1, 0, 1, 1)
            self.schedule_progress = QtWidgets.QProgressBar()
            progress_layout.addWidget(self.schedule_progress, 1, 1, 1, 1)
            self.schedule_label = QtWidgets.QLabel()
            self.schedule_label.setFixedWidth(RATE_COLUMN_WIDTH)
            self.schedule_label.setAlignment(Qt.AlignRight)
            progress_layout.addWidget(self.schedule_label, 1, 2, 1, 1)
            start = 3
        else:
            start = 2
        progress_layout.setRowMinimumHeight(
            start - 1, 20
        )  # space between the different sections
        for i, label in enumerate(self.queues):
            progress_layout.addWidget(QtWidgets.QLabel(label), i + start, 0, 1, 1)
            self.progress_bars[label] = QtWidgets.QProgressBar()
            self.progress_bars[label].setFormat("%v in queue")
            progress_layout.addWidget(self.progress_bars[label], i + start, 1, 1, 2)

        # General timing info
        progress_layout.setRowMinimumHeight(
            start + len(self.queues), 20
        )  # Add a bit of space between the different sections
        start += len(self.queues) + 1
        self.start_time = time.time()
        self.video_start_time = 0
        progress_layout.addWidget(
            QtWidgets.QLabel("Processing started:"), start, 0, 1, 1
        )
        progress_layout.addWidget(
            QtWidgets.QLabel(time.strftime("%H:%M:%S")), start, 1, 1, 1
        )
        progress_layout.addWidget(
            QtWidgets.QLabel("Elapsed time: "), start + 1, 0, 1, 1
        )
        self.elapsed_time = QtWidgets.QLabel()
        progress_layout.addWidget(self.elapsed_time, start + 1, 1, 1, 3)
        status_label = QtWidgets.QLabel("Status:")
        progress_layout.addWidget(status_label, start + 2, 0, 1, 1)
        self.status = QtWidgets.QLabel()
        progress_layout.addWidget(self.status, start + 2, 1, 1, 3)
        progress.setLayout(progress_layout)

        # Directories / space
        folders = QtWidgets.QGroupBox("Folders")
        folders_layout = QtWidgets.QVBoxLayout()
        self.source_dir = self.file_write_params["source_folder"]
        self.target_dir = self.file_write_params["target_folder"]

        self.source_dir_label = QtWidgets.QLabel(
            f"Source directory (free space: <b>{get_free_space(self.source_dir)}</b>):"
        )
        folders_layout.addWidget(self.source_dir_label)
        self.source_dir_link = QtWidgets.QLabel(f"<a href='#'>{self.source_dir}</a>")
        self.source_dir_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.source_dir_link.linkActivated.connect(lambda: open_dir(self.source_dir))
        folders_layout.addWidget(self.source_dir_link)

        self.target_dir_label = QtWidgets.QLabel(
            f"Target directory (free space: <b>{get_free_space(self.target_dir)}</b>):"
        )
        folders_layout.addWidget(self.target_dir_label)
        self.target_dir_link = QtWidgets.QLabel(f"<a href='#'>{self.target_dir}</a>")
        self.target_dir_link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.target_dir_link.linkActivated.connect(lambda: open_dir(self.target_dir))
        folders_layout.addWidget(self.target_dir_link)

        folders.setLayout(folders_layout)

        # Auto stop
        self.auto_stop = QtWidgets.QCheckBox("Stop automatically")
        self.auto_stop.setChecked(False)  # Be safe

        # Preview
        self.right = QtWidgets.QGroupBox("Image")
        self.orig_image = pg.ImageView()
        self.orig_image.setMinimumWidth(400)
        self.orig_image.ui.roiBtn.hide()
        self.orig_image.ui.menuBtn.hide()
        self.orig_image.getHistogramWidget().hide()
        self.processed_image = pg.ImageView()
        self.processed_image.ui.roiBtn.hide()
        self.processed_image.ui.menuBtn.hide()
        self.processed_image.getHistogramWidget().hide()
        pos = np.linspace(0, 1, 255)
        colors = [p for p in pos]  # list of colors (interpreted by pyqtgraph.mkColor)
        colors[-1] = MASK_COLOR
        cmap = pg.colormap.ColorMap(pos, colors)
        self.processed_image.setColorMap(cmap)
        self.orig_image.getImageItem().getViewBox().linkView(
            pg.ViewBox.XAxis, self.processed_image.getImageItem().getViewBox()
        )
        self.orig_image.getImageItem().getViewBox().linkView(
            pg.ViewBox.YAxis, self.processed_image.getImageItem().getViewBox()
        )

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.orig_image)
        right_layout.addWidget(self.processed_image)
        self.right.setLayout(right_layout)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.stop_button = QtWidgets.QPushButton("&Stop")
        self.stop_button.clicked.connect(self.stop)
        button_layout.addWidget(self.stop_button)

        self.quit_button = QtWidgets.QPushButton("&Quit")
        self.quit_button.clicked.connect(self.accept)
        button_layout.addWidget(self.quit_button)
        self.quit_button.setEnabled(False)

        main_layout = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        left.addWidget(progress)
        left.addWidget(folders)
        left.addWidget(self.auto_stop)
        left.addLayout(button_layout)
        main_layout.addLayout(left)
        main_layout.addWidget(self.right)

        self.setLayout(main_layout)

        self._max_tasks = 0

        # Scheduling support
        self.schedule = schedule
        self.recording = False
        self._current_schedule_duration = None
        self._last_schedule_switch = None

        # For runs without schedule, the epoch is always -1, and the index is the same as the original index
        self._epoch = -1
        self._epoch_start_idx = 0
        self.stopped = False

    def accept(self) -> None:
        self.quit_button.setEnabled(False)
        logger.info("Closing log")
        self.log_file.close()
        logger.removeHandler(self.log_file)
        return super().accept()

    def run(self):
        # self.create_overview_fig()  # TODO
        dirname = self.file_write_params["source_folder"]
        self.dir_observer = Observer()
        self.background_file_watcher = FileWatcher(self, dirname, self.fileno_offset, self.fileno_step)
        self.dir_observer.schedule(self.background_file_watcher, dirname)
        self.video_thread = None

        if self.record_video:
            self.video_thread = VideoThread(
                self,
                os.path.join(self.file_write_params["target_folder"], "video"),
                os.path.join(
                    self.file_write_params["target_folder"],
                    self.bg_params["filename_prefix"],
                ),
                task_queue=self.video_queue,
                file_queue=self.file_write_queue,
                delete_compressed=self.file_write_params["delete_compressed_files"],
                fps=self.file_write_params["fps"],
            )

        self.background_remover = YoloBackgroundRemover(
            progress_dialog=self,
            task_queue=self.processing_queue,
            bg_params=self.bg_params,
            file_write_params=self.file_write_params,
            inference_params=self.inference_params,
            video_queue=self.video_queue,
            file_queue=self.file_write_queue,
            link_tracks=self.link_tracks,
            track_settings=self.track_settings,
        )
        wait_for = 1
        if self.video_thread:
            wait_for += 1
        self.file_reader = FileReaderThread(
            parent=self,
            pool_size=MAX_FILE_READ_THREADS,
            frame_shape=self.bg_params["original_size"],
            read_queue=self.file_read_queue,
            processing_queue=self.processing_queue,
            write_queue=self.file_write_queue,
            read_function=self.read_function,
            delete_files=self.file_write_params["delete_files"],
        )
        self.file_writer = FileWriterThread(
            parent=self,
            task_queue=self.file_write_queue,
            wait_for=wait_for,
            **self.file_write_params,
        )

        self.background_file_watcher.file_available.connect(
            self.launch_read_file, type=Qt.QueuedConnection
        )

        self.file_reader.finished.connect(
            self.stop_processing, type=Qt.QueuedConnection
        )

        self.background_remover.preview_image.connect(
            self.preview_image, type=Qt.QueuedConnection
        )
        self.background_remover.finished.connect(
            self.file_writer.stop, type=Qt.QueuedConnection
        )

        if self.video_thread:
            self.video_thread.finished.connect(
                self.file_writer.stop, type=Qt.QueuedConnection
            )

        # Write a yaml file with the basic settings that were used
        roi_slice = self.bg_params["roi_slice"]
        settings = {
            "start": datetime.now(),
            "original_size": list(self.bg_params["original_size"]),
            "dark_field": self.bg_params["dark_field"],
            "roi_xy": [roi_slice[1].start, roi_slice[0].start],
            "roi_size": [
                roi_slice[1].stop - roi_slice[1].start,
                roi_slice[0].stop - roi_slice[0].start,
            ],        
            "video": {
                "enabled": self.record_video,
                "fps": self.file_write_params["fps"],
            },
            "filename_prefix": self.bg_params["filename_prefix"],
            "archive_compressed_files": self.archive_compressed_files,
            "read_function": self.read_function.__name__,
            "inference": self.inference_params,            
        }
        settings["inference"]["end2end"] = False
        if self.link_tracks:
            settings["tracking"] = {
                "link_tracks": True,
                **self.track_settings,
            }
            if settings["tracking"]["package"] == "yolo":
                tracker_config = os.path.join("config", self.track_settings["yolo"]["tracker_type"]+".yaml")
                with open(tracker_config) as f:
                    yolo_settings = yaml.safe_load(f)
                    settings["tracking"]["yolo"]["config_file"] = tracker_config
                    settings["tracking"]["yolo"].update(yolo_settings)
        
        settings.update(self.file_write_params)
        if self.schedule:
            # Use string representation for Timedelta objects
            settings.update({"schedule": {k: str(v) for k, v in self.schedule.items()}})
        os.makedirs(self.target_dir, exist_ok=True)
        with open(os.path.join(self.target_dir, "settings.yaml"), "wt") as f:
            yaml.dump(settings, f)

        # Write down settings in current directory as well
        with open(os.path.join(os.path.dirname(__file__), "last_yolo_settings.yaml"), "wt") as f:
            yaml.dump(settings, f)
        self.file_reader.setObjectName("FileReaderThread")
        self.file_reader.start()

        self.file_writer.setObjectName("FileWriterThread")
        self.file_writer.start()

        self.background_remover.setObjectName("BackgroundRemover")
        self.background_remover.start()

        if self.video_thread is not None:
            self.video_thread.setObjectName("VideoThread")
            self.video_thread.start()

        self.dir_observer.start()

        # Invoke initial run on separate thread
        class InitialRun(QtCore.QRunnable):
            def __init__(self, parent, watcher):
                super().__init__(parent=parent)
                self.watcher = watcher

            def run(self):
                self.watcher.initial_run()

        QtCore.QThreadPool.globalInstance().start(
            InitialRun(self, self.background_file_watcher)
        )

        # Update FPS calculation every half second
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_fps)
        self.timer.start(500)

        self.status.setText("Running")

    def check_schedule(self, fname, idx, ctime):
        # No schedule → record everything
        if not self.schedule:
            self.recording = True
            return

        # Discard all images that were created before the start of the program
        if ctime < self.start_time:
            self.background_remover.reset()
            self.recording = False
            logger.debug(
                f"Discarding image {idx} created before the start of the schedule"
            )
            return

        # We have a schedule
        schedule = self.schedule
        switched = False
        if self._last_schedule_switch is None:  # first image
            # For the first image, we apply the recording schedule without checking anything
            logger.info("First image, starting recording schedule")
            self.recording = True
            self._current_schedule_duration = schedule["record"].total_seconds()
            self._last_schedule_switch = time.time()
            self._epoch = 0
            self._epoch_start_idx = idx
            elapsed_since_last = 0
            self.schedule_progress.setMaximum(int(self._current_schedule_duration))
            switched = True
        else:
            # We are in the usual case, where we have to check the schedule
            elapsed_since_last = time.time() - self._last_schedule_switch
            if elapsed_since_last > self._current_schedule_duration:
                # We are past the current schedule
                logger.info(
                    f"Switching schedule : now {'no longer recording' if self.recording else 'recording'}"
                )

                if self.recording:
                    self.recording = False
                    self._current_schedule_duration = schedule[
                        "discard"
                    ].total_seconds()
                    # Request processing the current buffer
                    self.processing_queue.put(
                        (
                            idx
                            - 1
                            + 0.5,  # Make sure its gets processed before the first discarded image
                            {
                                "type": "stop",
                                "epoch": self._epoch,
                                "epoch_start_idx": self._epoch_start_idx,
                                "discard": False,
                                "final": False,
                            },
                        )
                    )
                else:
                    self.recording = True
                    self._current_schedule_duration = schedule["record"].total_seconds()
                    self._epoch += 1
                    self._epoch_start_idx = idx
                    self.background_remover.reset()

                self.schedule_progress.setMaximum(int(self._current_schedule_duration))
                self._last_schedule_switch = time.time()
                switched = True

        self.schedule_progress.setValue(int(elapsed_since_last))

        if switched:
            if self.recording:
                self.schedule_progress.setFormat(
                    f"Recording iteration {self._epoch+1} (%p%)"
                )
                self.schedule_progress.setPalette(
                    self.schedule_progress.style().standardPalette()
                )
            else:
                self.schedule_progress.setFormat("Not recording (%p%)")
                palette = QtGui.QPalette(self.schedule_progress.palette())
                palette.setColor(
                    QtGui.QPalette.Highlight, QtGui.QColor(QtCore.Qt.darkRed)
                )
                self.schedule_progress.setPalette(palette)

    def launch_read_file(self, fname, idx, ctime):
        self.check_schedule(fname, idx, ctime)
        task = {
            "type": "read",
            "fname": fname,
            "epoch": self._epoch,
            "relative_idx": idx - self._epoch_start_idx,
            "discard": not self.recording,
        }
        self.file_read_queue.put((idx, task))
        if self.recording:
            self.available_files += 1
        else:
            self.discarded_files += 1

    def stop(self):
        self.stopped = True
        self.stop_button.setEnabled(False)
        self.auto_stop.setEnabled(False)
        self.dir_observer.stop()
        # Tell file read queue to stop processing (but put it back in the queue)
        self.file_read_queue.put((sys.maxsize, {"type": "stop"}))

    def stop_processing(self):
        # Tell background remover to stop processing
        discard = not self.recording
        self.processing_queue.put(
            (
                sys.maxsize,
                {
                    "type": "stop",
                    "epoch": self._epoch,
                    "epoch_start_idx": self._epoch_start_idx,
                    "final": True,
                    "discard": discard,
                },
            )
        )
        # Fire off a thread that will wait for all threads to finish
        logger.info("Stopped processing, waiting for all other threads to finish")
        threads = [self.file_reader, self.background_remover, self.file_writer]
        if self.video_thread is not None:
            threads.append(self.video_thread)
        self.wait_thread = WaitThread(self, threads)
        self.wait_thread.setObjectName("WaitThread")
        self.wait_thread.finished.connect(self.all_done)
        self.wait_thread.start()

    def all_done(self):
        if exception_occured:
            self.status.setText(
                "Finished <span style='color:red'>(Exception occured, see log!)</span>"
            )
        else:
            self.status.setText("Finished")
        logger.debug("Verifying all queues are empty...")

        not_empty = []
        for name, task_queue in self.queues.items():
            if not task_queue.empty():
                not_empty.append(name)

        if not_empty:
            logger.error(f"Queues {not_empty} are not empty, this should not happen...")
            for q in not_empty:
                content = self.queues[q]
                logger.error(f"{q}: There are still {content.qsize()} entries:")
                for _ in range(content.qsize()):
                    logger.error(f"\t{content._queue.get_nowait()}")
        else:
            logger.info("All queues are empty, all good")
            for pb in self.progress_bars.values():
                pb.setValue(0)

        if self.archive_compressed_files:
            fnames = glob.glob(
                os.path.join(
                    self.file_write_params["target_folder"],
                    "frames",
                    self.bg_params["filename_prefix"] + "*.tiff",
                )
            )
            n_frames = len(fnames)
            if self.schedule:
                # Extract epochs from names
                epochs = max(set([int(os.path.basename(f).rsplit("_")[-2][-4:]) for f in fnames])) + 1
            else:
                epochs = 1

            # A bit lazy
            CODES = {
                1000: ("{chunk:04d}???.tiff", "{chunk:04d}xxx.zip"),
                10000: ("{chunk:03d}????.tiff", "{chunk:03d}xxxx.zip"),
                100000: ("{chunk:02d}?????.tiff", "{chunk:02d}xxxxx.zip"),
                1000000: ("{chunk:01d}??????.tiff", "{chunk:01d}xxxxxx.zip"),
            }
            if FRAMES_PER_ZIP not in CODES:
                logger.error(
                    f"Unsupported FRAMES_PER_ZIP value {FRAMES_PER_ZIP}, using 1000"
                )
                wildcard_code, zipfile_code = CODES[1000]
                n_chunks = int(np.ceil(n_frames / 1000 / epochs))
            else:
                wildcard_code, zipfile_code = CODES[FRAMES_PER_ZIP]
                n_chunks = int(np.ceil(n_frames / FRAMES_PER_ZIP / epochs))

            logger.info(f"Archiving {n_frames} frames into {n_chunks} zip archives")

            progress = QtWidgets.QProgressDialog(
                "Archiving files into zip archives..",
                "Abort archiving",
                0,
                n_chunks,
                self,
            )
            progress.setWindowModality(Qt.WindowModal)

            for epoch in range(epochs):
                for chunk in range(n_chunks):
                    progress.setValue(chunk + epoch * n_chunks)

                    if progress.wasCanceled():
                        logger.error(
                            f"Archiving was cancelled by user request after {chunk} archives"
                        )
                        break

                    # Use 7z to create archives
                    if self.schedule:
                        frame_wildcard = os.path.join(
                            self.file_write_params["target_folder"],
                            "frames",
                            self.bg_params["filename_prefix"]
                            + f"{epoch:04d}_"
                            + wildcard_code.format(chunk=chunk),
                        )
                        archive_name = os.path.join(
                        self.file_write_params["target_folder"],
                        "frames",
                        f"{self.bg_params['filename_prefix']}"
                        + f"{epoch:04d}_"
                        + zipfile_code.format(chunk=chunk),
                    )
                    else:
                        frame_wildcard = os.path.join(
                            self.file_write_params["target_folder"],
                            "frames",
                            self.bg_params["filename_prefix"]
                            + wildcard_code.format(chunk=chunk),
                        )
                        archive_name = os.path.join(
                            self.file_write_params["target_folder"],
                            "frames",
                            f"{self.bg_params['filename_prefix']}"
                            + zipfile_code.format(chunk=chunk),
                        )
                    arguments = [
                        "a",
                        "-tzip",
                        "-m0=bzip2",
                        "-sdel",
                        archive_name,
                        frame_wildcard,
                    ]
                    logger.debug(f"Archiving command: 7z {' '.join(arguments)}")
                    process = QtCore.QProcess()
                    process.start("7z", arguments)
                    process.waitForFinished(-1)
                    if process.exitCode() != 0:
                        logger.error(
                            f"Archiving failed with exit code {process.exitCode()}"
                        )
                        break

                progress.setValue(n_chunks)

        self.timer.stop()  # Stop updating the progress
        self.quit_button.setEnabled(True)

    def update_fps(self):
        global pause_reading
        # pause reading if memory is low
        if psutil.virtual_memory().percent > 90:
            logger.warning("Pausing reads (Memory almost full)")
            self.status.setText("Pausing reads <span style='color:red'>(Memory almost full)</span>")
            pause_reading = True
        else:
            run_text = "Finishing up" if self.stopped else "Running"
            if exception_occured:
                self.status.setText(
                    f"{run_text} <span style='color:red'>(Exception occured, see log!)</span>"
                )
            else:
                self.status.setText(run_text)
            pause_reading = False

        elapsed = time.time() - self.start_time

        if self.schedule:
            elapsed_since_last = time.time() - self._last_schedule_switch
            self.schedule_progress.setValue(int(elapsed_since_last))
            remaining = self._current_schedule_duration - elapsed_since_last
            if remaining > 0:
                self.schedule_label.setText(f"-{human_duration(remaining)}")
            else:
                self.schedule_label.setText("")

        # Stop if no more files are coming in
        if (
            self.stop_button.isEnabled()
            and self.last_ctime is not None
            and time.time() - self.last_ctime > 5
        ):
            logger.debug("No more files coming in the last five seconds")
            if self.auto_stop.isChecked():
                logger.info("Stopping reading in files due to auto stop feature")
                self.stop()
            else:
                logger.debug("Would stop, but auto stop is disabled")

        # Update progress bars with current tasks in queue
        tasks = []
        for q in self.queues.values():
            try:
                tasks.append(q._unfinished_tasks.get_value())
            except (AttributeError, NotImplementedError):
                try:
                    tasks.append(q.qsize())
                except NotImplementedError:  # on macOS
                    tasks.append(-1)
        max_tasks = max(tasks)
        if max_tasks == 0:
            max_tasks = 1
        for n_tasks, pb in zip(tasks, self.progress_bars.values()):
            if n_tasks == -1:
                pb.setMaximum(0)
            else:
                pb.setMaximum(max_tasks)
                pb.setValue(n_tasks)

        # Update FPS
        if self.available_files > self._last_available_files:
            self._available_files_per_s = f"{self.available_files/elapsed:>6.1f}"
            self._last_available_files = self.available_files
        else:
            self._available_files_per_s = "     ?"
            self._last_available_files = self.available_files

        if self.discarded_files > 0:
            self.source_file_total.setText(
                f"{self.available_files:6d} ({self.discarded_files:6d} discarded)"
            )
        else:
            self.source_file_total.setText(f"{self.available_files:6d}")
        self.source_file_rate.setText(f"{self._available_files_per_s}/s")

        # Update space
        self.source_dir_label.setText(
            f"Source directory (free space: <b>{get_free_space(self.source_dir)}</b>):"
        )
        self.target_dir_label.setText(
            f"Target directory (free space: <b>{get_free_space(self.target_dir)}</b>):"
        )
        # Update time
        self.elapsed_time.setText(human_duration(elapsed))

        if exception_occured:
            self.status.setText(
                "Running <span style='color:red'>(Exception occured, see log!)</span>"
            )

    def preview_image(self, fname, before_image, after_image):
        logger.debug("Showing new preview images")
        self.right.setTitle(f"Image ({fname})")
        has_image = self.orig_image.getImageItem().image is not None

        if has_image:
            # store previous zoom/pan
            orig_view_box = self.orig_image.getImageItem().getViewBox()
            orig_state = orig_view_box.getState()
            processed_view_box = self.processed_image.getImageItem().getViewBox()
            procsed_state = processed_view_box.getState()

        # update images
        self.orig_image.setImage(before_image)
        self.processed_image.setImage(after_image)

        if has_image:
            # restore zoom/pan
            orig_view_box.setState(orig_state)
            processed_view_box.setState(procsed_state)


class FindCellsWorker(QtCore.QThread):
    finished = QtCore.Signal()
    
    def __init__(self, queue: queue.LifoQueue):
        self.queue = queue        
        super().__init__()

    def run(self):
        while True:
            task = self.queue.get()
            if task is None:  # stop marker
                break
            model, image, conf, iou, initialize = (
                task["model"],
                task["image"],
                task["conf"],
                task["iou"],
                task["initialize"],
            )
            with torch.no_grad():
                results = model.predict(
                    [image],
                    imgsz=image.shape,
                    conf=conf,
                    iou=iou,
                    end2end=False,
                    predictor=OptimizedDetectionPredictor,
                    device=DEVICE,
                )
            image = torch.tensor(
                np.broadcast_to(image[None, None, :, :], (1, 3) + image.shape) / 255.0,
                dtype=torch.float32
            )
            post_results = extract_patches_centroid_theta(
                image[:, 0, :, :].to(device=results[0].boxes.xyxy.device),
                [results[0].boxes.xyxy],
            )
            self.boxes = post_results[0]["boxes_int"].cpu()
            self.mask = create_mask(self.boxes, image.shape[2:])
            self.confidence = results[0].boxes.conf.cpu().numpy()
            self.centroids = post_results[0]["centroid"].cpu().numpy()
            self.orientations = post_results[0]["orientation"].cpu().numpy()
            self.image = image
            self.initialize = initialize
            self.finished.emit()

# Inherit from Qt window
class FileCompressorGui(QtWidgets.QMainWindow):
    def __init__(self, directory=None):
        super().__init__(None)

        # Load settings from last run
        prev_settings = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), "last_yolo_settings.yaml"), "rt") as f:
                prev_settings = yaml.safe_load(f)
        except (FileNotFoundError, IOError, yaml.YAMLError) as ex:
            logger.warning(f"Could not load last settings: {ex}")            

        self.setWindowTitle("Yolo cell detection/tracking")
        self.resize(1000, 800)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        # Only display either source images or background image
        self.image_preview = pg.ImageView(discreteTimeLine=True)
        self.image_preview.ui.roiBtn.hide()
        self.roi_selector = None
        self.prev_size = None
        self.prev_roi_pos = None
        self.prev_roi_size = None
        self.inference_model = None
        self.masked_file_preview = None

        self.image_preview.ui.menuBtn.hide()
        self.image_preview.getHistogramWidget().hide()
        self.image_preview.sigTimeChanged.connect(lambda x, y: self.update_masked())

        self.masked_preview = pg.ImageView()        
        self.masked_preview.ui.roiBtn.hide()
        self.masked_preview.ui.menuBtn.hide()
        self.masked_preview.getHistogramWidget().hide()

        masked_preview_group = QtWidgets.QGroupBox()
        masked_preview_group.setTitle("Processed image")
        masked_preview_layout = QtWidgets.QVBoxLayout()
        masked_preview_layout.addWidget(self.masked_preview)
        masked_preview_group.setLayout(masked_preview_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.image_preview)
        splitter.addWidget(masked_preview_group)
        splitter.setSizes([1, 1])

        # Add panel for controls
        controls_layout = QtWidgets.QVBoxLayout()

        # Select source folder
        source_folder_group = QtWidgets.QGroupBox("Source images")
        source_folder_group_layout = QtWidgets.QVBoxLayout()
        source_folder_group.setLayout(source_folder_group_layout)
        layout = QtWidgets.QHBoxLayout()
        self.source_folder = QtWidgets.QLineEdit()
        self.source_folder.setPlaceholderText("Source folder")
        dir_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon
        )
        self.source_folder_button = QtWidgets.QPushButton(icon=dir_icon)
        self.source_folder_button.setToolTip("Select source folder")
        if prev_folder := prev_settings.get("source_folder", None):
            initial_dir = os.path.abspath(os.path.join(prev_folder, ".."))
        else:
            initial_dir = None
        self.source_folder_button.clicked.connect(
            lambda _, initial_dir=initial_dir: self.select_source_folder(initial_dir)
        )
        self.source_folder.editingFinished.connect(self.change_source_folder)
        layout.addWidget(self.source_folder)
        layout.addWidget(self.source_folder_button)
        source_folder_group_layout.addLayout(layout)
        controls_layout.addWidget(source_folder_group)

        self.files_label = QtWidgets.QLabel("No folder selected")
        self.files_label.setStyleSheet("QLabel {font-style: italic;}")
        self.source_file_size = None
        source_folder_group_layout.addWidget(self.files_label)
        self.dark_field_selected = QtWidgets.QCheckBox("&Dark field")
        self.dark_field = prev_settings.get("dark_field", False)
        self.dark_field_selected.setChecked(self.dark_field)
        self.dark_field_selected.stateChanged.connect(self.update_darkfield)
        source_folder_group_layout.addWidget(self.dark_field_selected)
        layout = QtWidgets.QHBoxLayout()
        read_library_label = QtWidgets.QLabel("TIFF reader:")
        self.read_library = QtWidgets.QComboBox()
        self.read_library.addItems(read_functions.keys())
        prev_read_function_name = prev_settings.get("read_function", "read_image_tifffile")
        for key, func in read_functions.items():
            if func.__name__ == prev_read_function_name:
                self.read_library.setCurrentText(key)
                break
        read_library_label.setBuddy(self.read_library)
        layout.addWidget(read_library_label)
        layout.addWidget(self.read_library)
        source_folder_group_layout.addLayout(layout)
        layout = QtWidgets.QHBoxLayout()
        self.schedule_label = QtWidgets.QLabel("S&chedule: <i>None</i>")
        file_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_FileIcon
        )
        schedule_button = QtWidgets.QPushButton(icon=file_icon)
        schedule_button.setToolTip("Select schedule file")
        schedule_button.clicked.connect(self.select_schedule)
        trash_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_TrashIcon
        )
        self.remove_schedule_button = QtWidgets.QPushButton(icon=trash_icon)
        self.remove_schedule_button.clicked.connect(self.remove_schedule)
        self.remove_schedule_button.setEnabled(False)
        self.schedule_label.setBuddy(schedule_button)
        layout.addWidget(self.schedule_label, stretch=2)
        layout.addWidget(schedule_button)
        layout.addWidget(self.remove_schedule_button)
        source_folder_group_layout.addLayout(layout)
        self.schedule = None

        # Select inference model
        model_group = QtWidgets.QGroupBox("Inference model")
        model_group_layout = QtWidgets.QVBoxLayout()
        model_group.setLayout(model_group_layout)
        layout = QtWidgets.QHBoxLayout()
        self.model_file = QtWidgets.QLineEdit()
        self.model_file.setPlaceholderText("Model weights file")
        self.model_file.setText(prev_settings.get("inference", {}).get("model_file", ""))
        file_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_FileIcon
        )
        self.model_button = QtWidgets.QPushButton(icon=file_icon)
        self.model_button.setToolTip("Select inference model file")
        self.model_button.clicked.connect(self.select_model_file)
        self.model_file.editingFinished.connect(self.change_model_file)
        layout.addWidget(self.model_file)
        layout.addWidget(self.model_button)
        model_group_layout.addLayout(layout)

        self.cell_label = QtWidgets.QLabel()
        model_group_layout.addWidget(self.cell_label)

        layout = QtWidgets.QHBoxLayout()
        batch_size_label = QtWidgets.QLabel("&Batch size: ")
        self.batch_size = QtWidgets.QSpinBox()
        self.batch_size.setMinimum(1)
        self.batch_size.setMaximum(1000)
        self.batch_size.setValue(prev_settings.get("inference", {}).get("batch_size", 1))
        self.batch_size.setKeyboardTracking(False)
        batch_size_label.setBuddy(self.batch_size)
        layout.addWidget(batch_size_label)
        layout.addWidget(self.batch_size)
        model_group_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        conf_threshold_label = QtWidgets.QLabel("&Confidence threshold: ")
        self.conf_threshold = QtWidgets.QDoubleSpinBox()
        self.conf_threshold.setMinimum(0.01)
        self.conf_threshold.setMaximum(1)
        self.conf_threshold.setSingleStep(0.1)
        self.conf_threshold.setValue(prev_settings.get("inference", {}).get("conf_threshold", 0.2))
        self.conf_threshold.setKeyboardTracking(False)
        self.conf_threshold.valueChanged.connect(lambda value: self.update_masked())
        conf_threshold_label.setBuddy(self.conf_threshold)
        layout.addWidget(conf_threshold_label)
        layout.addWidget(self.conf_threshold)
        model_group_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        iou_label = QtWidgets.QLabel("&IoU: ")
        self.iou = QtWidgets.QDoubleSpinBox()
        self.iou.setMinimum(0.01)
        self.iou.setMaximum(1)
        self.iou.setSingleStep(0.1)
        self.iou.setValue(prev_settings.get("inference", {}).get("iou", 0.7))
        self.iou.setKeyboardTracking(False)
        self.iou.valueChanged.connect(lambda value: self.update_masked())
        iou_label.setBuddy(self.iou)
        layout.addWidget(iou_label)
        layout.addWidget(self.iou)
        model_group_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()

        self.half_precision = QtWidgets.QCheckBox("&Half precision: ")        
        self.half_precision.setChecked(False)
        self.half_precision.setEnabled(False)
        self.half_precision.checkStateChanged.connect(lambda value: self.update_masked())
        layout.addWidget(self.half_precision)
        model_group_layout.addLayout(layout)

        self.use_tensorRT = QtWidgets.QCheckBox("Use &TensorRT: ")
        self._prev_use_tensorRT = prev_settings.get("inference", {}).get("tensorRT", False)
        self.use_tensorRT.setChecked(self._prev_use_tensorRT)
        self.use_tensorRT.checkStateChanged.connect(
            lambda value: setattr(self, "_prev_use_tensorRT", value)
        )
        layout.addWidget(self.use_tensorRT)
        model_group_layout.addLayout(layout)
        controls_layout.addWidget(model_group)

        tracking_group = QtWidgets.QGroupBox("Tracking")
        tracking_layout = QtWidgets.QVBoxLayout()
        tracking_group.setLayout(tracking_layout)
        link_track_layout = QtWidgets.QHBoxLayout()
        self.link_tracks = QtWidgets.QCheckBox("&Link tracks")
        self.link_tracks.setChecked("link_tracks" in prev_settings.get("tracking", {}) and prev_settings["tracking"]["link_tracks"])
        link_track_layout.addWidget(self.link_tracks)
        settings_icon = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView
        )
        self.track_settings_button = QtWidgets.QPushButton(icon=settings_icon)
        self.track_settings_button.setToolTip("Track settings")
        link_track_layout.addWidget(self.track_settings_button)
        self.default_track_settings = dict(DEFAULT_TRACK_SETTINGS)
        self.track_settings_button.clicked.connect(self.show_track_settings)

        self.track_settings = {'package': prev_settings.get("tracking", {}).get("package", self.default_track_settings['package'])}
        for package in self.default_track_settings['packages']:
            # Initialize with default settings
            self.track_settings[package] = {}
            for key in self.default_track_settings['packages'][package]:                
                self.track_settings[package][key] = self.default_track_settings['packages'][package][key].default
                if key in prev_settings.get("tracking", {}).get(package, {}):
                    self.track_settings[package][key] = prev_settings["tracking"][package][key]
        
        tracking_layout.addLayout(link_track_layout)

        layout = QtWidgets.QHBoxLayout()
        pixel_size_label = QtWidgets.QLabel("&Pixel size (µm): ")
        self.pixel_size = QtWidgets.QDoubleSpinBox()
        self.pixel_size.setMinimum(0.01)
        self.pixel_size.setMaximum(1000)
        self.pixel_size.setSingleStep(0.01)
        self.pixel_size.setValue(prev_settings.get("tracking", {}).get("pixel_size", 5.06))
        self.pixel_size.setKeyboardTracking(False)
        pixel_size_label.setBuddy(self.pixel_size)        

        layout.addWidget(pixel_size_label)
        layout.addWidget(self.pixel_size)
        tracking_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        self.movement_features = QtWidgets.QCheckBox("&Movement features")
        self.movement_features.setChecked(prev_settings.get("tracking", {}).get("movement_features", True))
        layout.addWidget(self.movement_features)

        self.zip_tracking_file = QtWidgets.QCheckBox("&Zip file")
        self.zip_tracking_file.setChecked(prev_settings.get("tracking", {}).get("zip_tracking_file", True))
        layout.addWidget(self.zip_tracking_file)

        tracking_layout.addLayout(layout)
        controls_layout.addWidget(tracking_group)

        if self.model_file.text():
            self.change_model_file()

        controls_layout.addStretch()

        # Select target folder and prefix
        target_group = QtWidgets.QGroupBox("Image saving")
        target_group_layout = QtWidgets.QVBoxLayout()
        target_group.setLayout(target_group_layout)

        layout = QtWidgets.QHBoxLayout()
        self.target_folder = QtWidgets.QLineEdit()
        self.target_folder.setPlaceholderText("Target folder")
        self.target_folder_button = QtWidgets.QPushButton(icon=dir_icon)
        self.target_folder_button.setToolTip("Select target folder")
        self.target_folder_button.clicked.connect(self.select_target_folder)
        layout.addWidget(self.target_folder)
        layout.addWidget(self.target_folder_button)
        target_group_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        prefix_label = QtWidgets.QLabel("File &prefix:")
        self.target_prefix = QtWidgets.QLineEdit()
        prefix_label.setBuddy(self.target_prefix)
        layout.addWidget(prefix_label)
        layout.addWidget(self.target_prefix)
        target_group_layout.addLayout(layout)

        # Select compression algorithm
        layout = QtWidgets.QHBoxLayout()
        compression_label = QtWidgets.QLabel("&Compression algorithm: ")
        self.compression_algorithm = QtWidgets.QComboBox()
        self.compression_algorithm.addItems(
            ["None", "lzw", "packbits", "deflate", "adobe_deflate", "lzma"]
        )
        self.compression_algorithm.setCurrentText(
           prev_settings.get("compression_algorithm", "packbits") 
        )  
        self.compression_algorithm.currentTextChanged.connect(
            self.update_target_file_size
        )
        compression_label.setBuddy(self.compression_algorithm)
        layout.addWidget(compression_label)
        layout.addWidget(self.compression_algorithm)
        target_group_layout.addLayout(layout)

        self.target_files_label = QtWidgets.QLabel()
        target_group_layout.addWidget(self.target_files_label)

        self.delete_files = QtWidgets.QCheckBox("&Delete original files")
        target_group_layout.addWidget(self.delete_files)

        self.delete_compressed_files = QtWidgets.QCheckBox("Delete &compressed files")
        target_group_layout.addWidget(self.delete_compressed_files)

        self.archive_compressed_files = QtWidgets.QCheckBox("&Archive compressed files")
        self.archive_compressed_files.setChecked(prev_settings.get("archive_compressed_files", True))
        target_group_layout.addWidget(self.archive_compressed_files)

        self.record_video = QtWidgets.QCheckBox("Write &video")
        self.record_video.setChecked(prev_settings.get("video", {}).get("enabled", True))
        target_group_layout.addWidget(self.record_video)

        def switch_delete_compressed_files():
            self.delete_compressed_files.setEnabled(self.record_video.isChecked())
            if not self.record_video.isChecked():
                self.delete_compressed_files.setChecked(False)
        self.record_video.stateChanged.connect(switch_delete_compressed_files)
        layout = QtWidgets.QHBoxLayout()
        fps_label = QtWidgets.QLabel("F&PS: ")
        self.framerate = QtWidgets.QDoubleSpinBox()
        self.framerate.setDecimals(1)
        self.framerate.setMinimum(1)
        self.framerate.setMaximum(120)
        # TODO: Take from image files
        self.framerate.setValue(prev_settings.get("video", {}).get("fps", 15))
        fps_label.setBuddy(self.framerate)
        layout.addWidget(fps_label)
        layout.addWidget(self.framerate)
        target_group_layout.addLayout(layout)

        controls_layout.addWidget(target_group)

        # Run button
        self.run_button = QtWidgets.QPushButton("Proceed")
        palette = self.run_button.palette()
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor("green"))
        self.run_button.setPalette(palette)
        self.run_button.setAutoFillBackground(True)
        self.run_button.clicked.connect(self.proceed)

        controls_layout.addWidget(self.run_button)

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(controls_layout, 0.5)
        self.layout.addWidget(splitter, 1.0)

        # Status bar
        status_bar = self.statusBar()
        self.status_text = QtWidgets.QLabel("")
        status_bar.addPermanentWidget(self.status_text, 1)
        self.progress_bar = QtWidgets.QProgressBar()
        status_bar.addPermanentWidget(self.progress_bar, 2)
        self.progress_bar.setVisible(False)

        self.fps_label = QtWidgets.QLabel("FPS: ?   ")
        status_bar.addPermanentWidget(self.fps_label, 0)

        self.central_widget.setLayout(self.layout)

        self.background = None
        self.bg_calc = None
        self.task_time = None
        self.task_items = 0
        self.file_number_timer = None
        self._from_automatic_threshold = False
        self._find_cells_queue = queue.LifoQueue(1)
        self._find_cells_worker = FindCellsWorker(self._find_cells_queue)
        self._find_cells_worker.finished.connect(self.cells_finished)
        self._find_cells_worker.start()

        # Set directory from command line
        if directory is not None:
            self.source_folder.setText(directory)
            self.source_folder.editingFinished.emit()

    def show_track_settings(self):
        dialog = QtWidgets.QDialog(parent=self)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog.setWindowTitle("Link settings")
        dialog.setModal(True)
        dialog.setLayout(QtWidgets.QVBoxLayout())
        # Combobox for selecting the tracking algorithm
        tracking_label = QtWidgets.QLabel("&Tracking algorithm")
        tracking_algorithm = QtWidgets.QComboBox()
        tracking_algorithm.addItems(self.default_track_settings['packages'].keys())
        tracking_algorithm.setCurrentText(self.track_settings['package'])
        tracking_label.setBuddy(tracking_algorithm)
        dialog.layout().addWidget(tracking_label)

        # Add group box for other settings
        settings_group = QtWidgets.QGroupBox("Settings")
        settings_layout = QtWidgets.QStackedLayout()
        setting_widgets = {}
        for package in self.default_track_settings['packages']:
            settings = self.default_track_settings['packages'][package]
            setting_values = self.track_settings[package]
            widget = SettingGUI(settings, setting_values)
            setting_widgets[package] = widget
            settings_layout.addWidget(widget)
        settings_group.setLayout(settings_layout)

        settings_layout.setCurrentIndex(tracking_algorithm.currentIndex())
        tracking_algorithm.activated.connect(settings_layout.setCurrentIndex)

        dialog.layout().addWidget(tracking_algorithm)
        dialog.layout().addWidget(settings_group)
        dialog.layout().addWidget(buttons)
        # Store settings if user accepts
        if dialog.exec():
            self.track_settings['package'] = tracking_algorithm.currentText()
            for package, setting_widget in setting_widgets.items():
                self.track_settings[package] = setting_widget.get_settings()

    def select_schedule(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select schedule file", filter="Schedule files (*.yaml)"
        )[0]
        if not fname:
            return
        try:
            with open(fname, "rt") as f:
                schedule = yaml.safe_load(f)
            if not len(schedule) == 1 and set(schedule.keys()) == {"schedule"}:
                raise ValueError("Schedule file needs to contain a 'schedule' element")
            schedule = schedule["schedule"]
            if not len(schedule) == 2 and set(schedule.keys()) == {
                "record",
                "duration",
            }:
                raise ValueError("The schedule needs to specify 'record' and 'discard'")
            schedule["record"] = pd.Timedelta(schedule["record"])
            schedule["discard"] = pd.Timedelta(schedule["discard"])
            self.schedule = schedule
        except Exception as ex:
            logger.error(f"Could not load schedule file '{fname}': {str(ex)}")
            QtWidgets.QErrorMessage(self).showMessage(
                f"Could not load schedule file '{fname}'\n\n {str(ex)}"
            )
            return
        self.schedule_label.setText(f"S&chedule: <i>{os.path.basename(fname)}</i>")
        self.remove_schedule_button.setEnabled(True)

    def remove_schedule(self):
        self.schedule = None
        self.schedule_label.setText("S&chedule: <i>None</i>")
        self.remove_schedule_button.setEnabled(False)

    def update_roi(self):
        self.update_masked()

    def update_darkfield(self):
        self.dark_field = self.dark_field_selected.isChecked()
        self.update_source_images()
        self.update_masked()

    def proceed(self):
        delete_files = self.delete_files.isChecked()

        if len(self.filenames) <= 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Insufficient files",
                "Need more than file to start processing",
            )
            return

        # Check whether the filenames contain numbers without gaps
        try:
            indices = np.array([extract_file_number(f) for f in sorted(self.filenames)])
        except TypeError:
            QtWidgets.QMessageBox.warning(
                self,
                "Filenames with invalid numbers",
                "Cannot extract a number from every filename",
            )
            return

        diffs = np.diff(indices[:-1])
        unique_diffs = np.unique(diffs)
        if len(unique_diffs) == 1 and unique_diffs[0] == 1:
            index_step = 0  # All good
        elif len(unique_diffs) > 1:
            QtWidgets.QMessageBox.warning(
                self,
                "Filenames with varying gaps",
                "The filename numbers are not consecutive, and the gaps are not consistent",
            )
            return
        else:  # consistent gaps (but not consecutive)
            index_step = unique_diffs[0]
            answer = QtWidgets.QMessageBox.question(
                self,
                "Non-consecutive filenames",
                f"The filename numbers are not consecutive, but follow each other with a gap of {index_step}. Continue by assuming this for all files?",
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        # Will open the progress dialog
        self.process_folder(delete_files, index_step)

    def preview_folder(self):
        if self.file_number_timer is not None:
            self.file_number_timer.stop()

        dirname = self.source_folder.text()

        self.filenames = get_image_fnames(dirname)
        if len(self.filenames) == 0:
            return

        n_frames = min([len(self.filenames), 50])
        read_function = read_functions[self.read_library.currentText()]

        # get first frame to determine size
        full_path = os.path.join(dirname, self.filenames[0])
        if os.path.splitext(full_path)[1] in [".tiff", ".tif"]:
            frame = read_function(full_path)
        else:
            frame = read_image_imageio(full_path)
        y, x = frame.shape

        self.start_task("Reading files", n_frames)
        total_size = 0
        self.preview_frames = []
        for idx, full_path in enumerate(self.filenames[:n_frames]):
            total_size += os.path.getsize(full_path)
            self.update_task(idx)
            if os.path.splitext(full_path)[1] in [".tiff", ".tif"]:
                frame = read_function(full_path)
            else:
                frame = read_image_imageio(full_path)
            self.preview_frames.append(frame)

        self.finish_task()

        self.source_file_size = total_size / n_frames
        filesize = human_filesize(self.source_file_size)
        self.files_label.setText(
            f"<b>{len(self.filenames)}</b> images in folder (<b>{filesize}</b>/file)"
        )

        self.target_prefix.setText(filename_prefix(self.filenames))

        # Update number of files every second with QTimer
        self.file_number_timer = QtCore.QTimer(self)
        self.file_number_timer.timeout.connect(self.update_file_number)
        self.file_number_timer.start(1000)

    def update_file_number(self):
        dirname = self.source_folder.text()
        self.filenames = get_image_fnames(dirname)
        filesize = human_filesize(self.source_file_size)
        self.files_label.setText(
            f"<b>{len(self.filenames)}</b> images in folder (<b>{filesize}</b>/file)"
        )

    def process_folder(self, delete_files=False, index_step=0):
        target_folder = self.target_folder.text()
        if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Target folder not empty",
                "The target folder is not empty. Please select an empty folder.",
            )
            return
        if self.archive_compressed_files.isChecked():
            # Check whether 7z is installed
            try:
                subprocess.check_call(
                    ["7z", "--help"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                QtWidgets.QMessageBox.warning(
                    self,
                    "7z not found",
                    "The 7z executable could not be found. Please install 7z from <a href='https://www.7-zip.org/'>www.7-zip.org</a>, and make sure it is in the PATH.",
                )
                return
        # Stop updating the number of files
        self.file_number_timer.stop()

        y, x = self.preview_frames[0].shape
        # Free the memory used for the initial frames
        del self.preview_frames
        self.preview_frames = None

        roi_slice = get_roi_slice(self.roi_selector)
        background_params = {
            "original_size": (y, x),
            "roi_slice": roi_slice,
            "dark_field": self.dark_field,
            "target_folder": target_folder,
            "filename_prefix": self.target_prefix.text(),
        }
        file_write_params = {
            "source_folder": self.source_folder.text(),
            "target_folder": target_folder,
            "compression_algorithm": self.compression_algorithm.currentText(),
            "delete_files": delete_files,
            "delete_compressed_files": self.delete_compressed_files.isChecked(),
            "fps": self.framerate.value(),
        }
        inference_params = {
            "model_file": self.model_file.text(),
            "conf_threshold": self.conf_threshold.value(),
            "iou": self.iou.value(),
            "half_precision": self.half_precision.isChecked(),
            "batch_size": self.batch_size.value(),             
            "tensorRT": self.use_tensorRT.isChecked(),
        }
        track_settings = dict(self.track_settings)
        track_settings.update({
            "pixel_size": self.pixel_size.value(),
            "movement_features": self.movement_features.isChecked(),
            "zip_tracking_file": self.zip_tracking_file.isChecked(),
        })
        dialog = ProgressDialog(
            self,
            bg_params=background_params,
            file_write_params=file_write_params,
            fileno_offset=self.fileno_offset,
            fileno_step=index_step,
            inference_params=inference_params,
            record_video=self.record_video.isChecked(),
            link_tracks=self.link_tracks.isChecked(),
            track_settings=track_settings,
            read_function=read_functions[self.read_library.currentText()],
            archive_compressed_files=self.archive_compressed_files.isChecked(),
            schedule=self.schedule,
        )

        # Preview thread is no longer needed
        self._find_cells_queue.put(None)
        self._find_cells_worker.deleteLater()

        dialog.run()
        exit_reason = dialog.exec()
        if exit_reason == 1:  # We interpret "accepted" as quit
            self.close()

    def update_source_images(self):
        images = np.array(self.preview_frames)
        if self.dark_field:
            images = 255 - images
        self.image_preview.setImage(images)
        self.image_preview.setCurrentIndex(0)
        if self.roi_selector is not None:
            prev_roi_pos = self.roi_selector.pos()
            prev_roi_size = self.roi_selector.size()
            self.image_preview.getView().removeItem(self.roi_selector)
        else:
            prev_roi_pos = prev_roi_size = None

        snap_size = 32

        if images[0].shape == self.prev_size and prev_roi_pos:
            pos = prev_roi_pos
            size = prev_roi_size
        else:
            height, width = images.shape[1], images.shape[2]
            excess_width, excess_height = width % snap_size, height % snap_size
            pos = (excess_width//2, excess_height//2)            
            size = width-excess_width, height-excess_height
            print("initial pos/size", pos, size)
        self.roi_selector = RectROI(
            pos,
            size,
            pen=(0, 9),
            snapSize=snap_size,
            translateSnap=True,
            scaleSnap=True,
            maxBounds=QtCore.QRectF(0, 0, images.shape[2], images.shape[1]),
        )
        self.roi_selector.addScaleHandle(pos=(0, 0), center=(1, 1))
        self.image_preview.getView().addItem(self.roi_selector)
        self.roi_selector.sigRegionChangeFinished.connect(lambda _: self.update_roi())

        # Remember for future changes
        self.prev_size = images[0].shape

    def update_masked(self, initialize=False):
        if not self.roi_selector:
            return

        current_idx = self.image_preview.currentIndex
        roi_slice = get_roi_slice(self.roi_selector)
        image = np.asarray(self.preview_frames[current_idx])
        image = image[roi_slice]

        # Switch back image but keep zoom/pan
        if not initialize:
            view_box = self.masked_preview.getImageItem().getViewBox()
            state = view_box.getState()
        self.masked_preview.setImage(image)
        if not initialize:
            view_box.setState(state)

        if self.inference_model:
            conf = self.conf_threshold.value()
            iou = self.iou.value()
            half_precision = self.half_precision.isChecked()
            self.start_task("Identifiying cells")            
            self._find_cells_queue.put({
                "image": image,
                "model": self.inference_model,
                "conf": conf,
                "iou": iou,
                "initialize": initialize,
            })
        else:
            self.cell_label.setText("")
            self.masked_file_preview = None
        self.target_files_label.setText("")

    @QtCore.Slot()
    def cells_finished(self):
        boxes = self._find_cells_worker.boxes
        confidence = self._find_cells_worker.confidence
        mask = self._find_cells_worker.mask
        image = self._find_cells_worker.image[0, 0, :, :].mul(255).to(dtype=torch.uint8)
        centroids = self._find_cells_worker.centroids
        orientations = self._find_cells_worker.orientations
        initialize = self._find_cells_worker.initialize
        if not initialize:
            view_box = self.masked_preview.getImageItem().getViewBox()
            state = view_box.getState()
        self.cell_label.setText(f"Found {len(boxes)} cells.")

        for child in self.masked_preview.getView().allChildren():
            # A bit dirty, but avoids us keeping a list of everything we add
            if isinstance(
                child,
                (
                    QtWidgets.QGraphicsRectItem,
                    QtWidgets.QGraphicsLineItem,
                    QtWidgets.QGraphicsTextItem,
                ),
            ):
                self.masked_preview.getView().removeItem(child)

        # Build an RGBA composite: original image + semi-transparent mask overlay
        rgba = torch.zeros((*image.shape, 4), dtype=torch.uint8)
        rgba[..., :3] = image[:, :, None]
        rgba[..., 3] = 255  # fully opaque base image
        # Overlay mask in semi-transparent red
        overlay = torch.zeros((*image.shape, 4), dtype=torch.uint8)
        overlay[mask] = torch.tensor([*MASK_COLOR, 120], dtype=torch.uint8)
        # Alpha-composite overlay onto rgba
        alpha = overlay[..., 3:4].to(dtype=torch.float32) / 255.0
        rgba[..., :3] = (
            overlay[..., :3] * alpha + rgba[..., :3] * (1 - alpha)
        ).to(dtype=torch.uint8)

        self.masked_preview.setImage(rgba.cpu().numpy())
        masked_file = image.cpu().numpy()
        masked_file[mask] = 255
        self.masked_file_preview = masked_file
        pen_color = QtGui.QColor("#C80000")
        pen_color.setAlpha(120)
        for (cx, cy), angle, confidence, (x1, y1, x2, y2) in zip(
            centroids, orientations, confidence, boxes
        ):
            square = QtWidgets.QGraphicsRectItem(x1, y1, x2-x1, y2-y1)            
            square.setPen(pg.mkPen(pen_color, width=1))

            self.masked_preview.getView().addItem(square)

            label = pg.TextItem(
                f"{confidence:.02f}", color=(200, 0, 0, 200), anchor=(0, 1)
            )
            label.setPos(x2, y1)
            self.masked_preview.getView().addItem(label)
            center_dot = QtWidgets.QGraphicsRectItem(cx-0.5, cy-0.5, 1, 1)
            center_dot.setPen(pg.mkPen(pen_color, width=1))
            center_dot.setBrush(pg.mkBrush(pen_color))
            self.masked_preview.getView().addItem(center_dot)
            radius = min(x2 - x1, y2 - y1) / 2
            lx1, ly1, lx2, ly2 = (
                cx + np.cos(angle) * radius,
                cy + np.sin(angle) * radius,
                cx - np.cos(angle) * radius,
                cy - np.sin(angle) * radius,
            )
            orientation_line = QtWidgets.QGraphicsLineItem(lx1, ly1, lx2, ly2)
            orientation_line.setPen(pg.mkPen(pen_color, width=2))
            self.masked_preview.getView().addItem(orientation_line)

        # Restore zoom/pan
        if not initialize:
            view_box.setState(state)
        self.update_target_file_size()
        self.finish_task()        

    def update_target_file_size(self):

        if self.masked_file_preview is None:
            self.target_files_label.setText("")
            return    
        in_memory_file = io.BytesIO()
        write_image(
            in_memory_file,
            self.masked_file_preview,
            compression=self.compression_algorithm.currentText(),
        )
        size = len(in_memory_file.getvalue())
        readable_size = human_filesize(size)
        factor = int(self.source_file_size / size)
        self.target_files_label.setText(
            f"Compressed: <b>{readable_size}</b> (factor ~<b>{factor}</b>)"
        )

    def select_source_folder(self, fallback_folder):
        start_dir = self.source_folder.text()
        if not start_dir:
            start_dir = fallback_folder
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select source folder", dir=start_dir
        )
        if folder:
            self.source_folder.setText(folder)
            self.source_folder.editingFinished.emit()

    def change_source_folder(self):
        folder = self.source_folder.text()
        if not folder:
            return        
        self.target_folder.setText(os.path.join(folder, "background_removed"))
        filenames = get_image_fnames(folder)
        n_files = len(filenames)

        if n_files == 0:
            self.files_label.setStyleSheet("QLabel {color: red;}")
        else:
            self.files_label.setStyleSheet("")
            # Extract filenumber from first filename to get general offset (usually 1 or 0)
            file_number = extract_file_number(filenames[0])
            if file_number is None:
                raise ValueError(
                    f"The file name '{filenames[0]}' does not end with a number"
                )
            self.fileno_offset = file_number
        self.preview_folder()
        self.update_source_images()
        self.update_masked(initialize=True)

    def select_model_file(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select model weights",
        )
        if fname:
            self.model_file.setText(fname)
            self.model_file.editingFinished.emit()

    def change_model_file(self):
        fname = self.model_file.text()
        self.inference_model = YOLO(fname)
        if fname.endswith(".engine"):
            _prev_use_tensorRT = self.use_tensorRT.isChecked()
            self.use_tensorRT.setChecked(True)
            self.use_tensorRT.setEnabled(False)
            self._prev_use_tensorRT = _prev_use_tensorRT
        else:
            if not self.use_tensorRT.isEnabled():
                self.use_tensorRT.setEnabled(True)
                self.use_tensorRT.setChecked(self._prev_use_tensorRT)
        self.update_masked()

    def select_target_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select target folder"
        )
        self.target_folder.setText(folder)

    def start_task(self, title, items=0):
        self.task_time = time.time()
        self.task_items = 0
        self.status_text.setText(title)
        self.progress_bar.setMaximum(items)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

    def update_task(self, item):
        self.progress_bar.setValue(item)
        elapsed = time.time() - self.task_time
        if elapsed > 1:
            self.fps_label.setText(f"FPS: {(item - self.task_items) / elapsed:.1f}  ")
            self.task_time = time.time()
            self.task_items = item
        elif self.task_items == 0:  # No FPS yet
            self.fps_label.setText("FPS: ?   ")

    def finish_task(self):
        self.status_text.setText("")
        self.progress_bar.setVisible(False)
        QtWidgets.QApplication.restoreOverrideCursor()

    @QtCore.Slot(int, str, np.ndarray)
    def frame_processed(self, idx, fname, frame):
        if idx >= 0:  # Ignore background frames
            self.update_task(idx)


if __name__ == '__main__':
    QLocale.setDefault(QLocale.C)  # do not use local decimal point settings
    if len(sys.argv) > 2:
        print(f"Ignoring arguments: '{' '.join(sys.argv[2:])}'", file=sys.stderr)
        print("Only a single directory argument is supported", file=sys.stderr, flush=True)
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        print(f"Setting directory: '{directory}'")
    else:
        directory = None

    app = QtWidgets.QApplication([])
    if platform.system == "Windows":
        app.setStyle('windowsvista')
    win = FileCompressorGui(directory)
    win.show()
    if os.environ.get("TEST_SHUTDOWN_GUI", "0") == "1":
        # Send close signal after one second
        QtCore.QTimer.singleShot(1000, app.quit)    
    app.exec()
