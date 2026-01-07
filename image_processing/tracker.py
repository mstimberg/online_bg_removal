import collections
from datetime import datetime
import time
import glob
import gzip
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import Process
from concurrent.futures import ThreadPoolExecutor
import concurrent
import os
import shutil
import warnings

import numpy as np
try:
    import cupy as cp
    cp.cuda.runtime.getDevice()
    xp = cp
except Exception:
    print("Tracker: Cupy/GPU not available, falling back to CPU")
    cp = None
    xp = np

import pandas as pd
import imageio.v3 as imageio

from image_processing.regionprops import determine_labels, determine_images, extract_properties
from image_processing.trajectory_analysis import segments_from_table, calculate_features, mark_avoiding_reactions_from_motion

TRACK_IN_PARALLEL = 8
TRIES = 10  # Number of times to try tracking a frame before giving up

CONSOLE_LOG_LEVEL = logging.WARNING
FILE_LOG_LEVEL = logging.INFO

# Make sure that log messages always have an index
def default_index_filter(record: logging.LogRecord):
    record.index = record.index if hasattr(record, "index") else -1
    return record

def log_namer(name):
    return name + ".gz"

def log_rotator(source, dest):
    with open(source, 'rb') as f_in:
        with gzip.open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(source)


class Tracker(Process):
    def __init__(self, pixel_size, min_area, max_area, target_folder, roi_size, features, track_tasks, link, zip_tracking_file, track_settings):
        super().__init__()
        self.pixel_size = pixel_size
        self.frame_rate = track_settings["frame_rate"]
        del track_settings["frame_rate"]
        self.min_area_pixels = min_area/pixel_size**2
        self.max_area_pixels = max_area/pixel_size**2
        self.target_folder = target_folder
        self.roi_size = roi_size
        self.link = link
        self.features = features
        self.zip_tracking_file = zip_tracking_file
        self.track_settings = track_settings
        props = ["bbox"]
        if 'ellipse' in features:
            props.append("ellipse")
        if 'orientation' in features:
            props.append("orientation")
        self.regionprops = tuple(props)
        self.nn_features = {}
        for feature in (set(features) - set(['ellipse', 'orientation', 'movement'])):
            try:
                from tensorflow.keras.models import load_model
            except ImportError:
                warnings.warn(f"Cannot calculate feature '{feature}', tensorflow/keras import failed")
                continue
            try:
                model = load_model(features[feature]["model-file"], custom_objects={'modified_mae' : None, 'mean_abs_difference_metric':None, 'combined_loss':None})
            except Exception as ex:
                warnings.warn(f"Could not load model for feature '{feature}': {ex}")
                continue
            column = features[feature]["column-name"]
            self.nn_features[feature] = {'model': model, "column": column}
        self.track_tasks = track_tasks
        self.columns = None
        self.cuda_stream = None
        self.start_time = None
        self.failure_counter = collections.defaultdict(int)
        self.logger = None

    def run(self):
        write_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TrackerWriteThread")
        self.logger = logging.getLogger(__file__)
        self.logger.setLevel(logging.DEBUG)
        _console_handler = logging.StreamHandler()
        _console_handler.setLevel(CONSOLE_LOG_LEVEL)
        _console_handler.setFormatter(logging.Formatter("%(asctime)s - %(threadName)s : %(levelname)s : %(message)s"))
        self.logger.addHandler(_console_handler)
        track_folder = os.path.join(self.target_folder, "tracking")
        os.makedirs(track_folder, exist_ok=True)
        log_fname = os.path.join(track_folder, f"log_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.log")
        log_file = RotatingFileHandler(log_fname, maxBytes=50_000_000, backupCount=10_000)
        log_file.rotator = log_rotator
        log_file.namer = log_namer
        log_file.setLevel(FILE_LOG_LEVEL)
        log_file.setFormatter(logging.Formatter("%(created)fs\t%(threadName)s\t%(levelname)s\t%(funcName)s:%(lineno)s\t%(index)d\t\"%(message)s\""))
        self.logger.addHandler(log_file)
        self.logger.addFilter(default_index_filter)
        self.logger.info("Tracker initialized")
        self.logger.info("Running with the following settings:")
        self.logger.info(f"Pixel size: {self.pixel_size:.2f} µm")
        self.logger.info(f"Frame rate: {self.frame_rate:.1f} fps")
        self.logger.info(f"Minimum area: {self.min_area_pixels:.0f} pixels")
        self.logger.info(f"Maximum area: {self.max_area_pixels:.0f} pixels")
        self.logger.info(f"Regionprops: {self.regionprops}")
        self.logger.info(f"Linking tracks: {self.link}")        

        stopped = False
        stop_after = None
        max_idx = -1
        self.start_time = time.time()
        processed = 0
        if cp:
            self.cuda_stream = cp.cuda.Stream(non_blocking=True)
        else:
            self.cuda_stream = None
        while not stopped or max_idx < stop_after:
            task = self.track_tasks.get()  # blocks until task is available

            if task['type'] == 'stop':
                epoch = task.get('epoch', -1)
                self.logger.info(f"Finishing up tracking tasks for epoch {epoch}")
                if not task.get('discard', False):
                    # make sure that all frames are processed
                    write_pool.shutdown(wait=True)
                    self.link_tracks(epoch, track_folder)
                self.track_tasks.task_done()
                if task['final']:
                    stopped = True
                    stop_after = task['last_idx']
                else:
                    # Create a new write pool
                    write_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TrackerWriteThread")
                continue

            assert task['type'] == 'track', task
            if task.get('discard', False):
                self.track_tasks.task_done()
                continue

            processed += 1

            try:
                self.track_frame_objects(task, track_folder, write_pool)
                self.track_tasks.task_done()
                max_idx = max(task['idx'], max_idx)
            except Exception:
                self.logger.exception("Error while tracking objects, will try again", extra={'index': task['idx']})
                self.failure_counter[task['idx']] += 1
                self.track_tasks.task_done()
                if self.failure_counter[task['idx']] < TRIES:
                    self.track_tasks.put(task)  # re-queue the task
                else:
                    self.logger.error(f"Failed to track frame {task['idx']} after {TRIES} tries, giving up")
                    max_idx = max(task['idx'], max_idx)

        if not self.track_tasks.empty():
            try:
                self.logger.error(f"There are still {self.track_tasks.qsize()} tasks in the queue")
            except NotImplementedError:
                self.logger.error(f"There are still tasks in the queue")

        self.logger.debug(
            f"Processed {processed} frames in {time.time() - self.start_time:.2f} seconds, i.e. {processed / (time.time() - self.start_time):.1f} frames per second"
        )

    def track_frame_objects(self, task, track_folder, write_pool):
        if self.cuda_stream:        
            with self.cuda_stream:  # do not block other CUDA operations
                idx, fname = task['idx'], task['fname']
                self.logger.debug(f"Reading frame from '{fname}'", extra={'index': idx})
                without_bg = cp.array(imageio.imread(fname))
                properties = self.track_objects(idx, without_bg)
        else:
            idx, fname = task['idx'], task['fname']
            self.logger.debug(f"Reading frame from '{fname}'", extra={'index': idx})
            without_bg = np.array(imageio.imread(fname))
            properties = self.track_objects(idx, without_bg)
        self.logger.debug("Writing properties to file", extra={'index': idx})
        df = pd.DataFrame(properties).rename(
            columns={
                "centroid-0": "y",
                "centroid-1": "x",
                "major_axis_length": "length",
                "minor_axis_length": "width",
                "orientation": "angle",
            }
        )
        if self.columns is None:  # Note down columns once
            self.columns = list(df.columns)
        write_pool.submit(self.write_properties, track_folder, idx, df)
        self.logger.debug("Wrote tracking data to 'tracking.tsv'", extra={'index': idx})
        return idx

    def write_properties(self, track_folder, idx, df):
        tracking_fname = f"frame_{idx:07d}.tsv"
        fname = os.path.join(track_folder, tracking_fname)
        df.to_csv(
            fname,
            columns=self.columns,
            mode="w",
            sep="\t",
            index=False,
            header=False,  # we don't write a header for easier merging later
            float_format="%.2f",
        )

    def apply_model(self, without_bg, model, input_size, properties, invert=True):        
        half_size = input_size // 2
        slices = []
        for r, c in zip(properties['centroid-0'], properties['centroid-1']):
            slice = xp.ones((input_size, input_size), dtype=without_bg.dtype)*np.iinfo(without_bg.dtype).max
            r1, r2 = max(0, int(r) - half_size), min(int(r) + half_size,  without_bg.shape[0])
            c1, c2 = max(0, int(c) - half_size), min(int(c) + half_size,  without_bg.shape[1])
            r_shift = input_size - (r2 - r1)
            c_shift = input_size - (c2 - c1)
            slice[r_shift // 2:r_shift // 2 + (r2 - r1), c_shift // 2:c_shift // 2 + (c2 - c1)] = without_bg[r1:r2, c1:c2]            
            if invert:
                slice = np.iinfo(without_bg.dtype).max - slice
            slices.append(slice)
        slice_stack = xp.stack(slices, axis=0)
        if cp:
            # Not that TF_FORCE_GPU_ALLOW_GROWTH needs to be set, otherwise Tensorflow will reserve all available memory
            # And no longer leave any space for cupy
            import tensorflow as tf
            slice_stack_dlpack = slice_stack.toDlpack()
            slice_stack = tf.experimental.dlpack.from_dlpack(slice_stack_dlpack)
        return model.predict(slice_stack[..., None])


    def track_objects(self, idx, without_bg):
        # 'labels' is just passed through to avoid re-allocating memory
        self.logger.debug("Tracking objects", extra={'index': idx})
        labels = determine_labels(without_bg, self.min_area_pixels, self.max_area_pixels)
        images, objects, indices_all = determine_images(labels)
        self.logger.debug("Extracting properties", extra={'index': idx})
        properties = extract_properties(idx, images, objects, indices_all, self.regionprops)
        for nn_feature, nn_data in self.nn_features.items():
            model = nn_data["model"]
            input_size = model.input_shape[1]
            model_prediction = self.apply_model(without_bg, model, input_size, properties)
            properties[nn_data["column"]] = np.squeeze(model_prediction)
        self.logger.debug("Finished tracking/determining properties", extra={'index': idx})
        return properties

    def link_tracks(self, epoch, track_folder):
        # Merge all files into one
        if epoch != -1:
            tracking_fname = os.path.join(track_folder, f"tracking_{epoch:07d}_unlinked_{self.frame_rate:.1f}_fps_{self.pixel_size:.2f}_um.tsv")
        else:
            tracking_fname = os.path.join(track_folder, f"tracking_unlinked_{self.frame_rate:.1f}_fps_{self.pixel_size:.2f}_um.tsv")
        if not os.path.exists(tracking_fname):
            self.logger.info(f"Linking tracks for epoch {epoch}, first concatenating all files")
            track_files = sorted(glob.glob(os.path.join(track_folder, "frame_*.tsv")))
            try:
                with open(tracking_fname, "w") as merged_file:
                    # Write header
                    merged_file.write("\t".join(self.columns) + "\n")
                    # Concatenate all files
                    for track_file in track_files:
                        with open(track_file, "r") as f:
                            shutil.copyfileobj(f, merged_file)
            except Exception:
                self.logger.exception(f"Error while merging tracking files for epoch {epoch}, keeping individual files")
                return
        
            self.logger.debug("Deleting individual files")
            for track_file in track_files:
                os.remove(track_file)
        else:
            self.logger.info(f"Linking tracks for epoch {epoch}, using existing file {tracking_fname}")

        if self.link:
            cells_df = pd.read_csv(tracking_fname, sep="\t")
            try:
                self.logger.info(f"Linking tracks for epoch {epoch}")
                linked = self.link_wrapper(cells_df)                
                if 'movement' in self.features:
                    segments = segments_from_table(linked)
                    self.logger.debug(f"Calculating features over {len(segments)} segments")
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        segments = executor.map(calculate_features, segments)
                        segments = executor.map(mark_avoiding_reactions_from_motion, segments)
                    linked = pd.concat(segments)
                    linked.sort_values(by='frame')
                    self.logger.debug("Finished calculating features")
                    if epoch != -1:
                        linked_fname = os.path.join(track_folder, f"tracking_{epoch:07d}_linked_with_features_{self.frame_rate:.1f}_fps_1_um.tsv")
                    else:
                        linked_fname = os.path.join(track_folder, f"tracking_linked_with_features_{self.frame_rate:.1f}_fps_1_um.tsv")
                else:
                    if epoch != -1:
                        linked_fname = os.path.join(track_folder, f"tracking_{epoch:07d}_linked_{self.frame_rate:.1f}_fps_1_um.tsv")
                    else:
                        linked_fname = os.path.join(track_folder, f"tracking_linked_{self.frame_rate:.1f}_fps_1_um.tsv")
                # ↑ Note that the filename states 1_um, since linked tracks are already scaled to µm
                if self.zip_tracking_file:
                    linked_fname += ".gz"
                linked.to_csv(linked_fname, sep="\t", index=False, float_format="%.2f")
                self.logger.info(f"linking tracks for epoch {epoch} done, linked tracks saved to {linked_fname}")
            except Exception:
                self.logger.exception(f"Linking tracks for epoch {epoch} failed")

    def link_wrapper(self, df):
        package = self.track_settings["package"]
        settings = dict(self.track_settings[package])
        # Preparations common to all packages
        # 1. Scale the coordinates and lengths to µm
        df[["x", "y", "bbox-0", "bbox-1", "bbox-2", "bbox-3"]] *= self.pixel_size
        if "length" in df.columns:
            df[["length", "width"]] *= self.pixel_size

        # 2. Convert search range and memory to µm and frames
        search_range = settings["maximum_speed"] / self.frame_rate
        memory = int(round(settings["memory"] * self.frame_rate))
        del settings["maximum_speed"]
        del settings["memory"]
        if package == "trackpy":
            import trackpy as tp
            if settings["adaptive_stop"] == 0:
                adaptive_stop = None
            else:
                adaptive_stop = settings["adaptive_stop"] / self.frame_rate
            del settings["adaptive_stop"]
            self.logger.info(f"Linking tracks with Trackpy, search_range={search_range}, memory={memory}, adaptive_stop={adaptive_stop}, {' '.join(f'{k}={v}' for k, v in settings.items())}")
            result = tp.link(df, search_range=search_range, memory=memory, adaptive_stop=adaptive_stop, **settings)
            result.rename(columns={"particle": "id"}, inplace=True)

        elif package == "norfair":
            import norfair
            initialization_delay = int(round(settings["initialization_delay"] * self.frame_rate))
            del settings["initialization_delay"]
            self.logger.info(f"Linking tracks with Norfair, distance_threshold={search_range}, hit_counter_max={memory}, initialization_delay={initialization_delay}, {' '.join(f'{k}={v}' for k, v in settings.items())}")
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
