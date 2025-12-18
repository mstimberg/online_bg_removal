"""
Script that watches a folder of files, removes the background and saves them as compressed files (optionally in a new
folder). The input file names need to have filenames ending in consecutive numbers.
"""

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
import skimage
import tifffile
import yaml
from pyqtgraph import RectROI
from PySide6.QtCore import Qt
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from image_processing.regionprops import (
    determine_images,
    determine_labels,
    extract_properties,
)
from image_processing.tracker import Tracker

DEFAULT_MIN_AREA = 400
DEFAULT_MAX_AREA = 12500

MASK_COLOR = "#FFC300"
ELLIPSE_COLOR = "#2219B2"
PREVIEW_EVERY = 100  # Show a preview image every X frames

MAX_FILE_READ_THREADS = 4

FFMPEG_VCODEC = "libx264"
FFMPEG_PRESET = "fast"
FFMPEG_PIX_FMT = "yuv420p"
CONSOLE_LOG_LEVEL = logging.WARNING
FILE_LOG_LEVEL = logging.INFO

MAX_FFMPEG_PROCESSES = 4  # Maximum number of ffmpeg processes to run in parallel

FRAMES_PER_ZIP = 1_000_000  # a single zip file...

try:
    import cupy as cp
    cp.cuda.runtime.getDevice()
except Exception:
    print("Background remover: Cupy/GPU not available, falling back to CPU")
    cp = None

# Use xp as a shorthand for the numpy-like library we are using
if cp:
    xp = cp
else:
    xp = np

# Class to store settings, docs, type, and min/max (or options) for each parameter
@dataclass
class Setting:
    """Class to store settings, docs, type, and min/max (or options) for each parameter"""
    name: str
    doc: str
    type: type
    unit: str = None
    str_evaluate: bool = False
    zero_as_none: bool = False  # If True, 0 is treated as None
    min: float = None
    max: float = None
    options: list = None
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


def write_image(path, array, pixelsize, compression=None):
    if not isinstance(path, str) or path.endswith(".tiff"):
        metadata = {
            "PhysicalSizeX": pixelsize,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixelsize,
            "PhysicalSizeYUnit": "µm",
        }
        resolution = (1e4 / pixelsize, 1e4 / pixelsize)
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
    return tuple(
        [
            slice(
                int(roi_selector.pos()[ax]),
                int(roi_selector.pos()[ax] + roi_selector.size()[ax]),
            )
            for ax in [1, 0]
        ]
    )


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
                logger.info(
                    "Stopping file reader thread, remaining tasks in queue: "
                    + str(self.read_queue.qsize())
                )
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


class BackgroundRemover(QtCore.QThread):
    frame_processed = QtCore.Signal(int, str, np.ndarray)
    preview_image = QtCore.Signal(str, np.ndarray, np.ndarray)

    def __init__(
        self,
        progress_dialog,
        task_queue,
        track_queue,
        background_frames,
        chunk_size,
        threshold,
        calc_every,
        initial_background,
        roi_slice,
        target_folder,
        filename_prefix,
        original_size,
        pixel_size,
        min_area,
        max_area,
        video_tasks,
        file_tasks,
        dark_field=False,
    ):
        super().__init__(parent=progress_dialog)
        self.progress_dialog = progress_dialog
        self.task_queue = task_queue
        self.track_queue = track_queue
        self.dark_field = dark_field
        self.background_frames = (background_frames,)
        self.chunk_size = chunk_size
        self.threshold = threshold
        self.calc_every = calc_every
        self.background = None
        self.pixel_size = pixel_size
        self.min_area_pixels = min_area / (pixel_size**2)
        self.max_area_pixels = max_area / (pixel_size**2)
        self.original_size = original_size
        self.bg_calc = BackgroundCalculator(
            background_frames,
            roi_slice[1].stop - roi_slice[1].start,
            roi_slice[0].stop - roi_slice[0].start,
            chunk_size,
            initial_background,
        )
        self.roi_slice = roi_slice
        self.target_folder = target_folder
        self.filename_prefix = filename_prefix
        self.video_tasks = video_tasks
        self.file_tasks = file_tasks
        self._last_idx = -1
        # We store the original file names for later
        self._orig_fnames = {}

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

            if task["type"] == "stop":  # stop processing the current buffer
                logger.info(
                    "Received sentinel item, finishing current buffer",
                    extra={"index": idx},
                )
                try:
                    self.finish_frames(
                        task["epoch"],
                        task["epoch_start_idx"],
                        task["final"],
                        task["discard"],
                    )
                    self._last_idx = idx
                    self.task_queue.task_done(index=idx)
                    if task.get("final", False):
                        # End thread
                        break
                    else:
                        if self.track_queue:
                            self.track_queue.put({"type": "stop", "final": False, "epoch": task["epoch"]})
                        continue
                except Exception:
                    logger.exception(
                        "Error while finishing frame buffer", extra={"index": idx}
                    )
                    self.task_queue.task_done(index=idx, measure=False)
                    raise

            assert task["type"] == "background", task

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
            frame = frame[self.roi_slice]
            if self.dark_field:
                frame = 255 - frame
            self.bg_calc.add_frame(frame)
            try:
                if (relative_idx + 1 == self.bg_calc.background_frames) or (
                    self.calc_every > 0
                    and (
                        (relative_idx - self.bg_calc.background_frames + 1)
                        % self.calc_every
                        == 0
                    )
                ):
                    logger.debug(
                        f"Calculating background for {self.bg_calc.background_frames} frames up to {idx}",
                        extra={"index": idx},
                    )
                    self.bg_calc.calc_background()
                    if epoch == -1:  # no schedule
                        bg_fname = f"backgrounds/background_uint8_{idx-self.bg_calc.background_frames+1:07}-{idx:07}.png"
                    else:
                        bg_fname = f"backgrounds/background_uint8_{epoch:04}_{relative_idx-self.bg_calc.background_frames+1:07}-{relative_idx:07}.png"
                    if cp:
                        bg_frame = self.bg_calc.background.get().astype(np.uint8)
                    else:
                        bg_frame = self.bg_calc.background.astype(np.uint8)
                    self.file_tasks.put(
                        {
                            "type": "frame",
                            "idx": -1,
                            "fname": bg_fname,
                            "frame": bg_frame,
                        }
                    )
            except Exception:
                logger.exception(
                    "Error while calculating background", extra={"index": idx}
                )
                self.task_queue.task_done(index=idx, measure=False)
                raise

            if self.bg_calc.chunk_size ==1 or (relative_idx > 0 and (relative_idx + 1) % self.bg_calc.chunk_size == 0):
                try:
                    self.remove_background(
                        self.bg_calc.chunk_size, frame, epoch, relative_idx
                    )
                    logger.debug("Removed background", extra={"index": idx})
                except Exception:
                    logger.exception(
                        "Error while removing background", extra={"index": idx}
                    )
                    self.task_queue.task_done(index=idx, measure=False)
                    raise

            self.task_queue.task_done(index=idx)

    def remove_background(self, n_frames, last_frame, epoch, relative_idx_start):
        logger.debug(
            f"Removing background for {n_frames} frames, up to {self._last_idx}",
            extra={"index": self._last_idx},
        )
        if n_frames == 0:
            return
        # Remove background for all frames in buffer at once
        removed = self.bg_calc.remove_backgrounds(self.threshold, n_frames=n_frames)
        if self.video_tasks is not None:
            logger.debug(
                f"Background remover: submitted video task for frames up to {self._last_idx}",
                extra={"index": self._last_idx},
            )
            self.video_tasks.put(
                {
                    "idx": self._last_idx,
                    "n_frames": n_frames,
                    "epoch": epoch,
                    "relative_start_idx": relative_idx_start,
                    "discard": False,
                }
            )
        if cp:
            removed = removed.get()
        else:
            removed = removed.copy()
        for counter, processed_frame in enumerate(removed):
            frame_idx = self._last_idx - n_frames + counter + 1
            relative_idx = relative_idx_start - n_frames + counter + 1
            # We start our file names with 1 for ffmpeg
            if epoch == -1:
                filename = os.path.join(
                    "frames", self.filename_prefix + f"{frame_idx + 1:07d}.tiff"
                )
            else:
                filename = os.path.join(
                    "frames",
                    self.filename_prefix + f"{epoch:04d}_{relative_idx + 1:07d}.tiff",
                )
            logger.debug(
                f"Background remover: processed frame {frame_idx} ('{filename}')",
                extra={"index": frame_idx},
            )
            self.file_tasks.put(
                {
                    "type": "frame",
                    "idx": frame_idx,
                    "fname": filename,
                    "frame": processed_frame,
                }
            )
        if last_frame is not None:
            self.preview_image.emit(filename, last_frame, processed_frame)

    def finish_frames(self, epoch, relative_idx_start, final, discard):
        if not discard:
            n_frames = (
                self._last_idx - relative_idx_start + 1
            ) % self.bg_calc.chunk_size

            # Remove background for the last frames in the buffer
            self.remove_background(
                n_frames, None, epoch, self._last_idx - relative_idx_start
            )

        # Clear buffer
        self.bg_calc.frame_index = 0
        self.bg_calc.calculated = False

        if self.video_tasks is not None:
            # Request merging of video files
            self.video_tasks.put(
                {"stop": True, "final": final, "epoch": epoch, "discard": discard}
            )


class FileWriterThread(QtCore.QThread):
    def __init__(
        self,
        parent,
        source_folder,
        target_folder,
        compression_algorithm,
        pixel_size,
        fps,
        task_queue,
        track_queue,
        wait_for=1,
        delete_files=False,
        delete_compressed_files=False,
    ):
        super().__init__(parent=parent)
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.compression_algorithm = compression_algorithm
        self.pixel_size = pixel_size
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
            pixelsize=self.pixel_size,
        )
        logger.debug(f"Wrote '{fname}' (with imageio)", extra={"index": idx})
        self.task_queue.task_done(index=idx)
        if self.track_queue is not None and idx >= 0:
            logger.debug(
                "FileWriter: submitted tracking task for file '{fname}'",
                extra={"index": idx},
            )
            self.track_queue.put({"type": "track", "idx": idx, "fname": full_path})

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


class BackgroundCalculator:
    def __init__(
        self,
        background_frames,
        width,
        height,
        chunk_size,
        initial_background,
        dtype=np.uint8,
    ):
        self.background_frames = background_frames
        self.dtype = dtype
        self.buffer = xp.zeros((chunk_size, height, width), dtype=dtype)
        self.diff_images = xp.zeros((chunk_size, height, width), dtype=xp.uint8)
        self.background_sum = xp.zeros((height, width), dtype=xp.int32)
        self.background = xp.array(initial_background, dtype=dtype)

        self.frame_index = 0
        self.calculated = False
        self.chunk_size = chunk_size

    def add_frame(self, frame):
        frame_on_gpu = xp.asarray(frame, dtype=self.dtype)
        self.background_sum += frame_on_gpu

        self.buffer[self.frame_index] = frame_on_gpu
        self.frame_index += 1

    def calc_background(self):
        self.background[:] = xp.asarray(
            self.background_sum / self.background_frames, dtype=self.dtype
        )
        self.background_sum[:] = 0

    def remove_backgrounds(self, threshold, n_frames=None):
        """
        Remove background from the last n_frames in the buffer at once
        """
        if n_frames is None:
            n_frames = self.chunk_size
        frames = self.buffer[:n_frames]        
        background_inv = xp.clip(
            xp.invert(self.background).astype("int16") + threshold, 0, 255
        ).astype("uint8")
        # Use in-place operations to save memory
        xp.clip(
            xp.invert(frames), background_inv, None, out=self.diff_images[:n_frames]
        )
        self.diff_images[:n_frames] -= background_inv
        np.invert(self.diff_images[:n_frames], out=self.diff_images[:n_frames])
        self.frame_index = 0
        return xp.asarray(self.diff_images[:n_frames], dtype=self.dtype)


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
        logger.debug(
            f"{self._name}: Adding task to queue with size {self._queue.qsize()}, counter: {self.counter}"
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
    def __init__(self, parent, threads, tracker_process, track_queue, epoch):
        super().__init__(parent=parent)
        self.threads = threads
        self.tracker_process = tracker_process
        self.track_queue = track_queue
        self.epoch = epoch

    def run(self):
        threading.current_thread().name = QtCore.QThread.currentThread().objectName()
        if TRACE:
            get_tracer().enable_thread_tracing()

        # Wait for all given threads to finish
        for thread in self.threads:
            thread.wait()

        if self.tracker_process is not None:
            self.track_queue.put({"type": "stop", "final": True, "last_idx": 0 ,"epoch": self.epoch})
            while self.tracker_process.is_alive():
                logger.debug("Waiting for tracker process to finish")
                self.tracker_process.join(10)
            logger.info("Tracker process has finished")

        logger.info("WaitThread finished")

class ProgressDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent,
        bg_params,
        file_write_params,
        fileno_offset,
        fileno_step,
        track,
        track_features,
        link_tracks,
        zip_tracking_file,
        track_settings,
        record_video,
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
        self.track = track
        self.track_features = track_features        
        self.link_tracks = link_tracks        
        self.zip_tracking_file = zip_tracking_file
        self.track_settings = track_settings
        self.record_video = record_video
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
        self.track_queue = None
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
        if track:
            self.track_queue = JoinableQueue()
            self.queues["Tracking"] = self.track_queue

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

    def create_overview_fig(self):
        # We create a basic overview figure with matplotlib showing the elements of the processing
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, axs = plt.subplots(
            2, 2, layout="constrained", figsize=(1920 // 72, 1080 // 72)
        )
        # Original image
        if cp:
            source_image = self.bg_calc.circular_buffer[0].get()
        else:
            source_image = self.bg_calc.circular_buffer[0]
        axs[0, 0].imshow(source_image, cmap="gray")
        axs[0, 0].add_patch(
            Rectangle(
                (self.roi_slice[1].start, self.roi_slice[0].start),
                self.roi_slice[1].stop - self.roi_slice[1].start,
                self.roi_slice[0].stop - self.roi_slice[0].start,
                edgecolor="darkred",
                facecolor="none",
                lw=2,
            )
        )
        axs[0, 0].set_title("Source image")

        # ROI region
        axs[0, 1].imshow(source_image[self.roi_slice], cmap="gray")
        axs[0, 1].set_title("Region of interest")

        bg_calc = self.bg_params["bg_calc"]
        # Background
        if cp:
            axs[1, 0].imshow(bg_calc.background.get(), cmap="gray")
        else:
            axs[1, 0].imshow(bg_calc.background, cmap="gray")
        axs[1, 0].set_title(f"Background (average of {bg_calc.buffersize})")

        removed = bg_calc.remove_background(
            source_image, bg_calc.background, self.threshold
        )
        axs[1, 1].imshow(removed, cmap="gray")
        axs[1, 1].set_title(f"Final image (thresholded at {self.threshold})")

        for idx, ax in enumerate(axs.flat):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if idx > 0:
                ax.spines["bottom"].set_color("darkred")
                ax.spines["top"].set_color("darkred")
                ax.spines["right"].set_color("darkred")
                ax.spines["left"].set_color("darkred")

        fig.savefig(
            os.path.join(self.target_dir, "backgrounds", "overview.tiff"), dpi=72
        )

    def run(self):
        # self.create_overview_fig()  # TODO
        dirname = self.file_write_params["source_folder"]
        self.dir_observer = Observer()
        self.background_file_watcher = FileWatcher(self, dirname, self.fileno_offset, self.fileno_step)
        self.dir_observer.schedule(self.background_file_watcher, dirname)
        self.tracker = None

        self.video_thread = None

        if self.track:
            self.tracker = Tracker(
                pixel_size=self.bg_params["pixel_size"],
                min_area=self.bg_params["min_area"],
                max_area=self.bg_params["max_area"],
                target_folder=self.target_dir,
                roi_size=(
                    self.bg_params["roi_slice"][0].stop
                    - self.bg_params["roi_slice"][0].start,
                    self.bg_params["roi_slice"][1].stop
                    - self.bg_params["roi_slice"][1].start,
                ),
                features=self.track_features,
                link=self.link_tracks,
                zip_tracking_file=self.zip_tracking_file,
                track_settings=self.track_settings,
                track_tasks=self.track_queue,
            )

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

        self.background_remover = BackgroundRemover(
            progress_dialog=self,
            task_queue=self.processing_queue,
            track_queue=self.track_queue,
            **self.bg_params,
            video_tasks=self.video_queue,
            file_tasks=self.file_write_queue,
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
            track_queue=self.track_queue,
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
            "using_cupy": cp is not None,
            "original_size": list(self.bg_params["original_size"]),
            "dark_field": self.bg_params["dark_field"],
            "roi_xy": [roi_slice[1].start, roi_slice[0].start],
            "roi_size": [
                roi_slice[1].stop - roi_slice[1].start,
                roi_slice[0].stop - roi_slice[0].start,
            ],
            "background": {
                "type": "fixed_average_before",
                "thresholding": "shifted_background",
                "frames": self.bg_params["background_frames"],
                "recalculated_every": self.bg_params["calc_every"],
            },
            "processing": {
                "chunk_size": self.bg_params["chunk_size"],
            },
            "tracking": {"enabled": self.track},
            "video": {
                "enabled": self.record_video,
                "fps": self.file_write_params["fps"],
            },
            "filename_prefix": self.bg_params["filename_prefix"],
            "archive_compressed_files": self.archive_compressed_files,
            "threshold": self.bg_params["threshold"],
            "read_function": self.read_function.__name__,
        }
        if self.track:
            settings["tracking"].update(
                {
                    "pixel_size": self.bg_params["pixel_size"],
                    "min_area": self.bg_params["min_area"],
                    "max_area": self.bg_params["max_area"],
                    "zip_tracking_file": self.zip_tracking_file,
                    "needs_axis_swap": False,
                }
            )
            settings["tracking"].update(self.track_features)
            if self.link_tracks:
                settings["tracking"]["link_tracks"] = True
                settings["tracking"]["package"] = self.track_settings["package"]
                settings["tracking"][self.track_settings["package"]] = self.track_settings[self.track_settings["package"]]
        settings.update(self.file_write_params)
        if self.schedule:
            # Use string represetation for Timedelta objects
            settings.update({"schedule": {k: str(v) for k, v in self.schedule.items()}})
        os.makedirs(self.target_dir, exist_ok=True)
        with open(os.path.join(self.target_dir, "settings.yaml"), "wt") as f:
            yaml.dump(settings, f)

        # Write down settings in current directory as well
        with open(os.path.join(os.path.dirname(__file__), "last_settings.yaml"), "wt") as f:
            yaml.dump(settings, f)
        self.file_reader.setObjectName("FileReaderThread")
        self.file_reader.start()

        self.file_writer.setObjectName("FileWriterThread")
        self.file_writer.start()

        self.background_remover.setObjectName("BackgroundRemover")
        self.background_remover.start()

        if self.tracker is not None:
            self.tracker.start()  # separate process

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
            self.background_remover.bg_calc.frame_index = 0
            self.background_remover.bg_calc.caclulated = False
            self.background_remover.bg_calc.background_sum[:] = 0
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
                    self.background_remover.bg_calc.frame_index = 0
                    self.background_remover.bg_calc.caclulated = False
                    self.background_remover.bg_calc.background_sum[:] = 0

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
        self.wait_thread = WaitThread(self, threads, self.tracker, self.track_queue, self._epoch)
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
        for task_queue in self.queues.values():
            if not task_queue.empty():
                not_empty.append(task_queue._name)

        if not_empty:
            logger.error(f"Queues {not_empty} are not empty, this should not happen...")
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
        tasks = [
            getattr(getattr(q, "_unfinished_tasks", None), "get_value", q.qsize)()
            for q in self.queues.values()
        ]
        max_tasks = max(tasks)
        if max_tasks == 0:
            max_tasks = 1
        for n_tasks, pb in zip(tasks, self.progress_bars.values()):
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
        self.orig_image.setImage(before_image.T)
        self.processed_image.setImage(after_image.T)

        if has_image:
            # restore zoom/pan
            orig_view_box.setState(orig_state)
            processed_view_box.setState(procsed_state)


# Inherit from Qt window
class FileCompressorGui(QtWidgets.QMainWindow):
    def __init__(self, directory=None):
        super().__init__(None)

        # Load settings from last run
        prev_settings = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), "last_settings.yaml"), "rt") as f:
                prev_settings = yaml.safe_load(f)
        except (FileNotFoundError, IOError, yaml.YAMLError) as ex:
            logger.warning(f"Could not load last settings: {ex}")            

        self.feature_settings = {}
        try:
            with open(os.path.join(os.path.dirname(__file__), "config", "features.yaml"), "rt") as f:
                self.feature_settings = yaml.safe_load(f)
        except (FileNotFoundError, IOError, yaml.YAMLError) as ex:
            logger.exception("Could not load feature settings")

        self.setWindowTitle("Background removal settings")
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

        self.image_preview.ui.menuBtn.hide()
        self.image_preview.getHistogramWidget().hide()
        self.image_preview.sigTimeChanged.connect(lambda x, y: self.update_subtracted())

        self.background_preview = pg.ImageView()
        self.background_preview.ui.roiBtn.hide()
        self.background_preview.ui.menuBtn.hide()
        self.background_preview.getHistogramWidget().hide()

        self.image_tab = QtWidgets.QTabWidget()
        self.image_tab.addTab(self.image_preview, "Source images")
        self.image_tab.addTab(self.background_preview, "Background image")

        self.subtracted_preview = pg.ImageView()
        self.subtracted_preview.getImageItem().getViewBox().sigRangeChanged.connect(
            self.update_ellipses
        )
        self.subtracted_preview.ui.roiBtn.hide()
        self.subtracted_preview.ui.menuBtn.hide()
        self.subtracted_preview.getHistogramWidget().hide()

        subtracted_preview_group = QtWidgets.QGroupBox()
        subtracted_preview_group.setTitle("Processed image")
        subtracted_preview_layout = QtWidgets.QVBoxLayout()
        subtracted_preview_layout.addWidget(self.subtracted_preview)
        subtracted_preview_group.setLayout(subtracted_preview_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(self.image_tab)
        splitter.addWidget(subtracted_preview_group)

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
        self.source_folder_button.clicked.connect(self.select_source_folder)
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

        background_group = QtWidgets.QGroupBox("Background removal")
        background_group_layout = QtWidgets.QVBoxLayout()
        background_group.setLayout(background_group_layout)

        # Select number of background frames
        layout = QtWidgets.QHBoxLayout()
        background_frames_label = QtWidgets.QLabel("&Background frames: ")
        self.background_frames = QtWidgets.QSpinBox()
        self.background_frames.setMinimum(1)
        self.background_frames.setMaximum(1000)
        self.background_frames.setKeyboardTracking(False)
        self.background_frames.setValue(prev_settings.get("background", {}).get("frames", DEFAULT_BUFFERSIZE))
        self.background_frames.valueChanged.connect(lambda x: self.set_buffersize(x))

        layout.addWidget(background_frames_label)
        layout.addWidget(self.background_frames)
        background_frames_label.setBuddy(self.background_frames)
        background_group_layout.addLayout(layout)

        # When to update background
        layout = QtWidgets.QHBoxLayout()
        background_update_label = QtWidgets.QLabel("&Update every: ")
        self.background_update = MulitiplierSpinBox(self, n_values=100)
        self.background_update.setValue(prev_settings.get("background", {}).get("recalculated_every", 1)//self.background_frames.value())
        self.background_update.lineEdit().setReadOnly(True)
        background_update_label.setBuddy(self.background_update)
        layout.addWidget(background_update_label)
        layout.addWidget(self.background_update)
        background_group_layout.addLayout(layout)

        # Block size for processing
        layout = QtWidgets.QHBoxLayout()
        block_size_label = QtWidgets.QLabel("&Block size: ")
        self.block_size = DivisorSpinBox(self, max_value=self.background_frames.value())
        self.block_size.lineEdit().setReadOnly(True)
        prev_chunk_size = prev_settings.get("processing", {}).get("chunk_size", self.background_frames.value())
        chunk_size_idx = np.argmin(np.abs(np.array(self.block_size.divisors) - prev_chunk_size))
        self.block_size.setValue(chunk_size_idx + 1)
        block_size_label.setBuddy(self.block_size)
        layout.addWidget(block_size_label)
        layout.addWidget(self.block_size)
        background_group_layout.addLayout(layout)

        # Select threshold
        layout = QtWidgets.QHBoxLayout()
        threshold_label = QtWidgets.QLabel("T&hreshold: ")
        self.threshold = QtWidgets.QSpinBox()
        self.threshold.setMinimum(0)
        self.threshold.setMaximum(255)
        self.threshold.setValue(prev_settings.get("threshold", 0))
        self.threshold.setEnabled(False)
        self.threshold.setKeyboardTracking(False)
        self.threshold.valueChanged.connect(self.update_threshold)
        layout.addWidget(threshold_label)
        layout.addWidget(self.threshold)
        threshold_label.setBuddy(self.threshold)

        background_group_layout.addLayout(layout)

        # Automatic threshold determination
        layout = QtWidgets.QHBoxLayout()
        self.automatic_threshold_selected = QtWidgets.QCheckBox("&Automatic")
        self.automatic_threshold_selected.stateChanged.connect(
            lambda _: self.automatic_threshold()
        )
        self.automatic_threshold_selected.setEnabled(False)
        self.automatic_threshold_algo = QtWidgets.QComboBox()
        self.automatic_threshold_algo.addItems(
            ["Isodata", "Li", "Mean", "Minimum", "Otsu", "Triangle", "Yen"]
        )
        self.automatic_threshold_algo.setCurrentText("Yen")
        self.automatic_threshold_algo.currentTextChanged.connect(
            lambda _: self.automatic_threshold()
        )
        layout.addWidget(self.automatic_threshold_selected)
        layout.addWidget(self.automatic_threshold_algo)
        background_group_layout.addLayout(layout)
        controls_layout.addWidget(background_group)

        controls_layout.addStretch()

        # Segmentation/tracking
        tracking_group = QtWidgets.QGroupBox("Tracking")
        tracking_layout = QtWidgets.QVBoxLayout()
        tracking_group.setLayout(tracking_layout)

        layout = QtWidgets.QHBoxLayout()
        pixel_size_label = QtWidgets.QLabel("&Pixel size (µm): ")
        self.pixel_size = QtWidgets.QDoubleSpinBox()
        self.pixel_size.setMinimum(0.01)
        self.pixel_size.setMaximum(1000)
        self.pixel_size.setSingleStep(0.01)
        self.pixel_size.setValue(prev_settings.get("pixel_size", 5.06))
        self.pixel_size.setKeyboardTracking(False)
        self.pixel_size.valueChanged.connect(self.update_pixel_size)
        pixel_size_label.setBuddy(self.pixel_size)

        layout.addWidget(pixel_size_label)
        layout.addWidget(self.pixel_size)
        tracking_layout.addLayout(layout)

        layout = QtWidgets.QHBoxLayout()
        track_area_label = QtWidgets.QLabel("Min/max area (µm²): ")
        self.min_track_area = QtWidgets.QSpinBox()
        self.min_track_area.setMinimum(1)
        self.min_track_area.setMaximum(1000000)  # will be set to max_area value later
        self.min_track_area.setSingleStep(10)
        self.min_track_area.setValue(prev_settings.get("tracking", {}).get("min_area", DEFAULT_MIN_AREA))
        self.min_track_area.setKeyboardTracking(False)
        self.min_track_area.valueChanged.connect(self.update_pixel_size)

        self.max_track_area = QtWidgets.QSpinBox()        
        self.max_track_area.setMaximum(1000000)
        self.max_track_area.setSingleStep(100)
        self.max_track_area.setValue(prev_settings.get("tracking", {}).get("max_area", DEFAULT_MAX_AREA))        
        self.max_track_area.setKeyboardTracking(False)
        self.max_track_area.valueChanged.connect(self.update_pixel_size)
        self.min_track_area.setMaximum(self.max_track_area.value() - 1)
        self.max_track_area.setMinimum(self.min_track_area.value() + 1)
        layout.addWidget(track_area_label)
        layout.addWidget(self.min_track_area)
        layout.addWidget(self.max_track_area)
        tracking_layout.addLayout(layout)

        self.track_cells = QtWidgets.QCheckBox("&Track cells")
        self.track_cells.setChecked(prev_settings.get("tracking", {}).get("enabled", False))
        self.track_cells.stateChanged.connect(self.update_tracking)
        tracking_layout.addWidget(self.track_cells)

        self._track_labels = []
        self._track_ellipses = []

        self.feature_checkboxes = {}
        for feature, feature_desc in self.feature_settings.items():
            cb = QtWidgets.QCheckBox(feature_desc["label"])
            cb.setToolTip(feature_desc["doc"])            
            cb.setChecked(feature in prev_settings.get("tracking", {}))
            cb.stateChanged.connect(self.update_feature_checkboxes)
            self.feature_checkboxes[feature] = cb

        self.zip_tracking_file = QtWidgets.QCheckBox("&Zip file")
        self.zip_tracking_file.setChecked(prev_settings.get("tracking", {}).get("zip_tracking_file", True))
        if not self.track_cells.isChecked():
            self.zip_tracking_file.setEnabled(False)        

        link_track_layout = QtWidgets.QHBoxLayout()
        self.link_tracks = QtWidgets.QCheckBox("&Link tracks")
        self.link_tracks.setChecked("link_tracks" in prev_settings.get("tracking", {}) and prev_settings["tracking"]["link_tracks"])
        if not self.track_cells.isChecked():
            self.link_tracks.setEnabled(False)
        self.link_tracks.stateChanged.connect(self.update_feature_checkboxes)
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

        for feature_cb in self.feature_checkboxes.values():
            tracking_layout.addWidget(feature_cb)
        tracking_layout.addWidget(self.zip_tracking_file)
        tracking_layout.addLayout(link_track_layout)

        controls_layout.addWidget(tracking_group)

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

        # Set directory from command line
        if directory is not None:
            self.source_folder.setText(directory)
            self.source_folder.editingFinished.emit()

    def update_feature_checkboxes(self):
        selected = {
            "track": self.track_cells.isChecked(),
            "link": self.link_tracks.isChecked(),
        }
        for feature, feature_cb in self.feature_checkboxes.items():
            selected[feature] = feature_cb.isChecked()
        for feature, feature_desc in self.feature_settings.items():
            for dep in feature_desc["dependencies"]:
                if not selected[dep]:
                    self.feature_checkboxes[feature].setChecked(False)
                    self.feature_checkboxes[feature].setEnabled(False)
                    break
            else:  # we did not break out of the for loop – all dependencies are fulfilled
                self.feature_checkboxes[feature].setEnabled(True)

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

    def set_threshold_from_histogram(self, x):
        # A bit of a hacky workaround
        if not self._from_automatic_threshold:
            self.automatic_threshold_selected.setChecked(False)
        self.threshold.setValue(x.getLevels()[1])

    def update_threshold(self):
        if not self._from_automatic_threshold:
            self.automatic_threshold_selected.setChecked(False)
        self.update_subtracted()
        self.update_target_file_size()
        self.update_tracking_preview()

    def update_ellipses(self):
        if not len(self._track_ellipses):
            return
        view_box = self.subtracted_preview.getImageItem().getViewBox()
        view_range = view_box.state["viewRange"]
        offset_x = view_range[0][1] - (view_range[0][1] - view_range[0][0]) / 2
        offset_y = view_range[1][1] - (view_range[1][1] - view_range[1][0]) / 2
        min_ellipse, max_ellipse = self._track_ellipses
        max_ellipse.setRect(
            offset_x - max_ellipse.rect().width() / 2,
            offset_y - max_ellipse.rect().height() / 2,
            max_ellipse.rect().width(),
            max_ellipse.rect().height(),
        )
        min_ellipse.setRect(
            offset_x
            + (max_ellipse.rect().width() - min_ellipse.rect().width()) / 2
            - max_ellipse.rect().width() / 2,
            offset_y
            + (max_ellipse.rect().height() - min_ellipse.rect().height()) / 2
            - max_ellipse.rect().height() / 2,
            min_ellipse.rect().width(),
            min_ellipse.rect().height(),
        )

    def update_roi(self):
        self.update_subtracted()
        self.update_tracking_preview()

    def update_pixel_size(self):
        self.min_track_area.setMaximum(self.max_track_area.value() - 1)
        self.max_track_area.setMinimum(self.min_track_area.value() + 1)
        self.update_tracking_preview()

    def update_darkfield(self):
        self.dark_field = self.dark_field_selected.isChecked()
        self.update_source_images()
        self.update_background()
        self.update_subtracted()

    def update_tracking(self):
        if not self.track_cells.isChecked():
            self.link_tracks.setEnabled(False)
            self.delete_compressed_files.setEnabled(True)
        else:
            self.link_tracks.setEnabled(True)
            # The tracking process needs the files
            self.delete_compressed_files.setEnabled(False)
            self.delete_compressed_files.setChecked(False)
        self.update_feature_checkboxes()
        self.update_tracking_preview()

    def update_tracking_preview(self):
        for label in self._track_labels:
            self.subtracted_preview.getView().removeItem(label)
        for ellipses in self._track_ellipses:
            self.subtracted_preview.getView().removeItem(ellipses)
        self._track_labels.clear()
        self._track_ellipses.clear()
        self.track_cells.setText("Track cells")
        if (
            self.track_cells.isChecked()
            and self.subtracted_preview.getImageItem().image is not None
        ):
            image = xp.array(self.subtracted_preview.getImageItem().image)
            min_area_pixels = self.min_track_area.value() / (self.pixel_size.value() ** 2)
            max_area_pixels = self.max_track_area.value() / (self.pixel_size.value() ** 2)
            # Visualize area by plotting ellipses with a 3/1 ratio
            view_box = self.subtracted_preview.getImageItem().getViewBox()
            view_range = view_box.state["viewRange"]
            offset_x = view_range[0][1] - (view_range[0][1] - view_range[0][0]) / 2
            offset_y = view_range[1][1] - (view_range[1][1] - view_range[1][0]) / 2
            pen_color = QtGui.QColor(ELLIPSE_COLOR)
            pen_color.setAlpha(200)
            max_ellipse_width = 2 * np.sqrt(max_area_pixels / (3 * np.pi))
            max_ellipse_height = max_ellipse_width * 3
            max_area_ellipse = QtWidgets.QGraphicsEllipseItem(
                offset_x - max_ellipse_width / 2,
                offset_y - max_ellipse_height / 2,
                max_ellipse_width,
                max_ellipse_height,
            )
            max_area_ellipse.setPen(pg.mkPen(pen_color, width=3))
            self.subtracted_preview.getView().addItem(max_area_ellipse)
            min_ellipse_width = 2 * np.sqrt(min_area_pixels / (3 * np.pi))
            min_ellipse_height = min_ellipse_width * 3
            min_area_ellipse = QtWidgets.QGraphicsEllipseItem(
                offset_x
                + (max_ellipse_width - min_ellipse_width) / 2
                - max_ellipse_width / 2,
                offset_y
                + (max_ellipse_height - min_ellipse_height) / 2
                - max_ellipse_height / 2,
                min_ellipse_width,
                min_ellipse_height,
            )
            min_area_ellipse.setPen(pg.mkPen(pen_color, width=2))
            self.subtracted_preview.getView().addItem(min_area_ellipse)
            self._track_ellipses = [min_area_ellipse, max_area_ellipse]
            labels = determine_labels(image, min_area_pixels, max_area_pixels)
            images, objects, indices_all = determine_images(labels)
            properties = extract_properties(
                0, images, objects, indices_all, regionprops=()
            )

            if not len(properties):
                self.track_cells.setText("Track cells (no cells found)")
                return
            labeled = sorted(
                [
                    (c, r)
                    for r, c in zip(properties["centroid-0"], properties["centroid-1"])
                ]
            )
            for idx, (x, y) in enumerate(labeled):
                label = pg.TextItem(
                    f"{idx+1}", color=(200, 0, 0, 200), anchor=(0.5, 0.5)
                )
                label.setPos(y, x)
                self._track_labels.append(label)
                self.subtracted_preview.getView().addItem(label)
            self.track_cells.setText(f"Track cells ({len(labeled)} cells found)")

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

    def automatic_threshold(self):
        active = self.automatic_threshold_selected.isChecked()
        if active:
            self._from_automatic_threshold = True
            self.apply_automatic_threshold()
            self._from_automatic_threshold = False

    def preview_folder(self):
        if self.file_number_timer is not None:
            self.file_number_timer.stop()

        dirname = self.source_folder.text()
        n_frames = self.background_frames.value()

        self.filenames = get_image_fnames(dirname)
        if len(self.filenames) == 0:
            return

        if len(self.filenames) < n_frames:
            n_frames = len(self.filenames)
            self.background_frames.setValue(n_frames)
            self.background_frames.setMaximum(n_frames)

        self.threshold.setEnabled(True)
        self.automatic_threshold_selected.setEnabled(True)

        read_function = read_functions[self.read_library.currentText()]

        # get first frame to determine size
        full_path = os.path.join(dirname, self.filenames[0])
        if os.path.splitext(full_path)[1] in [".tiff", ".tif"]:
            frame = read_function(full_path)
        else:
            frame = read_image_imageio(full_path)
        y, x = frame.shape

        self.bg_calc = InitialBackgroundCalculator(x, y)

        self.start_task("Reading files", n_frames)
        total_size = 0
        self.bg_calc.resize(n_frames)
        for idx, full_path in enumerate(self.filenames[:n_frames]):
            total_size += os.path.getsize(full_path)
            self.update_task(idx)
            if os.path.splitext(full_path)[1] in [".tiff", ".tif"]:
                frame = read_function(full_path)
            else:
                frame = read_image_imageio(full_path)
            self.bg_calc.add_frame(frame)

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
        self.background_frames.setMaximum(len(self.filenames))

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

        x, y = self.bg_calc.frames[0].shape
        # Free the memory used for the initial frames
        del self.bg_calc
        self.bg_calc = None
        calc_every = (
            int(self.background_update.text())
            if self.background_update.value() > 0
            else 0
        )
        roi_slice = get_roi_slice(self.roi_selector)
        background_params = {
            "original_size": (x, y),
            "background_frames": self.background_frames.value(),
            "threshold": self.threshold.value(),
            "calc_every": calc_every,
            "chunk_size": self.block_size.divisors[self.block_size.value() - 1],
            "initial_background": self.initial_background[roi_slice],
            "roi_slice": roi_slice,
            "dark_field": self.dark_field,
            "target_folder": target_folder,
            "filename_prefix": self.target_prefix.text(),
            "pixel_size": self.pixel_size.value(),
            "min_area": self.min_track_area.value(),
            "max_area": self.max_track_area.value(),
        }
        file_write_params = {
            "source_folder": self.source_folder.text(),
            "target_folder": target_folder,
            "compression_algorithm": self.compression_algorithm.currentText(),
            "delete_files": delete_files,
            "delete_compressed_files": self.delete_compressed_files.isChecked(),
            "pixel_size": self.pixel_size.value(),
            "fps": self.framerate.value(),
        }
        # Track settings also need the frame rate
        self.track_settings['frame_rate'] = self.framerate.value()
        dialog = ProgressDialog(
            self,
            background_params,
            file_write_params,
            self.fileno_offset,
            index_step,
            track=self.track_cells.isChecked(),
            track_features={
                f: (
                    {
                        "model-file": self.feature_settings[f]["model-file"],
                        "column-name": self.feature_settings[f]["column-name"],
                    }
                    if "model-file" in self.feature_settings[f]
                    else True
                )
                for f, f_cb in self.feature_checkboxes.items()
                if f_cb.isChecked()
            },
            link_tracks=self.link_tracks.isChecked(),
            zip_tracking_file=self.zip_tracking_file.isChecked(),
            track_settings=self.track_settings,
            record_video=self.record_video.isChecked(),
            schedule=self.schedule,
            read_function=read_functions[self.read_library.currentText()],
            archive_compressed_files=self.archive_compressed_files.isChecked(),
        )

        dialog.run()
        exit_reason = dialog.exec()
        if exit_reason == 1:  # We interpret "accepted" as quit
            self.close()

    def set_buffersize(self, n_frames):
        # Force an update
        self.background_update.setValue(self.background_update.value())
        self.block_size.update_divisors(n_frames)
        if self.bg_calc is not None and n_frames > len(self.bg_calc.frames):
            # Read new frames to fill up the buffer
            read_function = read_functions[self.read_library.currentText()]
            self.start_task("Reading files", n_frames - len(self.bg_calc.frames))
            self.bg_calc.resize(n_frames)
            for idx, full_path in enumerate(
                self.filenames[self.bg_calc._n_frames : n_frames]
            ):
                self.update_task(idx)
                if os.path.splitext(full_path)[1] in [".tiff", ".tif"]:
                    frame = read_function(full_path)
                else:
                    frame = read_image_imageio(full_path)
                self.bg_calc.add_frame(frame)
            self.finish_task()
        self.update_source_images()
        self.update_background()
        self.update_subtracted()

    def update_source_images(self):
        if self.bg_calc is None or self.bg_calc.frames.shape[0] == 0:
            # Folder without images
            self.image_preview.clear()
            return

        images = self.bg_calc.frames.transpose(0, 2, 1)
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

        if images[0].shape == self.prev_size and prev_roi_pos:
            pos = prev_roi_pos
            size = prev_roi_size
        else:
            pos = (0, 0)
            size = (images.shape[1], images.shape[2])
        self.roi_selector = RectROI(
            pos,
            size,
            pen=(0, 9),
            snapSize=16,
            translateSnap=True,
            scaleSnap=True,
            maxBounds=QtCore.QRectF(0, 0, images.shape[1], images.shape[2]),
        )
        self.roi_selector.addScaleHandle(pos=(0, 0), center=(1, 1))
        self.image_preview.getView().addItem(self.roi_selector)
        self.roi_selector.sigRegionChangeFinished.connect(lambda _: self.update_roi())

        # Remember for future changes
        self.prev_size = (
            self.image_preview.getImageItem().width(),
            self.image_preview.getImageItem().height(),
        )

    def update_background(self):
        if self.bg_calc is None:
            # Folder without images
            self.background_preview.clear()
            return
        self.start_task("Calculating background")
        background = self.initial_background = self.bg_calc.calc_background(
            self.background_frames.value()
        )
        self.finish_task()
        if self.dark_field:
            background = 255 - self.initial_background
        self.background_preview.setImage(background.T)

    def update_subtracted(self, initialize=False):
        if self.bg_calc is None:
            self.subtracted_preview.clear()
            return

        current_idx = self.image_preview.currentIndex
        roi_slice = get_roi_slice(self.roi_selector)

        image = self.bg_calc.frames[current_idx]
        image = image[roi_slice]
        background = self.initial_background[roi_slice]
        if self.dark_field:
            image = 255 - image
            background = 255 - background

        threshold = self.threshold.value()
        subtracted = self.bg_calc.remove_background(image, background, threshold).T
        # Store zoom/pan
        if not initialize:
            view_box = self.subtracted_preview.getImageItem().getViewBox()
            state = view_box.getState()
            threshold = self.threshold.value()

        self.subtracted_preview.setImage(subtracted)
        if self.automatic_threshold_selected.isChecked():
            self.apply_automatic_threshold()

        self.update_target_file_size()

        pos = np.linspace(0, 1, 255)
        colors = [p for p in pos]  # list of colors (interpreted by pyqtgraph.mkColor)
        colors[-1] = MASK_COLOR
        cmap = pg.colormap.ColorMap(pos, colors)
        self.subtracted_preview.setColorMap(cmap)

        # Restore zoom/pan
        if not initialize:
            view_box.setState(state)
            if not self.automatic_threshold_selected.isChecked():
                self.threshold.setValue(threshold)
        
        # Update tracked cells
        if self.track_cells.isChecked():
            self.update_tracking_preview()

    def update_target_file_size(self):
        subtracted = np.clip(
            np.asarray(self.subtracted_preview.image, dtype=np.uint8),
            0,
            self.threshold.value(),
        )
        in_memory_file = io.BytesIO()
        write_image(
            in_memory_file,
            subtracted,
            compression=self.compression_algorithm.currentText(),
            pixelsize=self.pixel_size.value(),
        )
        size = len(in_memory_file.getvalue())
        readable_size = human_filesize(size)
        factor = int(self.source_file_size / size)
        self.target_files_label.setText(
            f"Compressed: <b>{readable_size}</b> (factor ~<b>{factor}</b>)"
        )

    def apply_automatic_threshold(self):
        algo = self.automatic_threshold_algo.currentText()
        func = getattr(skimage.filters, f"threshold_{algo.lower()}")
        threshold = int(
            round(
                func(
                    np.clip(
                        self.image_preview.image[0], self.background_preview.image, None
                    ).astype("int16")
                    - self.background_preview.image
                )
            )
        )
        self.threshold.setValue(threshold)

    def select_source_folder(self):
        start_dir = self.source_folder.text()
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
        self.bg_calc = None  # Make sure old results are no longer around
        self.target_folder.setText(os.path.join(folder, "background_removed"))
        filenames = get_image_fnames(folder)
        n_files = len(filenames)
        self.background_frames.setMaximum(n_files)
        if self.background_frames.value() == 0:
            self.background_frames.setValue(min(n_files, DEFAULT_BUFFERSIZE))
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
        self.update_background()
        self.update_subtracted(initialize=True)
        self.update_tracking()

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
        self.progress_bar.setVisible(True)
        # TODO: Disable controls

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

    @QtCore.Slot(int, str, np.ndarray)
    def frame_processed(self, idx, fname, frame):
        if idx >= 0:  # Ignore background frames
            self.update_task(idx)


if __name__ == '__main__':
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
