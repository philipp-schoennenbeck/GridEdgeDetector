import math
import sys, os
from collections import namedtuple
from PyQt5 import QtCore
try:
    from cryovia.gui import starting_menu
except:
    pass
from matplotlib import pyplot as plt
import subprocess, os, platform
import traceback
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from PyQt5.QtCore import QAbstractTableModel, Qt, QSize, QThread, QObject, pyqtSignal, QModelIndex, QItemSelectionModel,QTimer
import shutil
import pickle
from PyQt5.QtGui import QImage, QPixmap, QColor, QIntValidator, QDoubleValidator, QPen, QValidator, QPalette, QKeySequence, QWheelEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QStyledItemDelegate, QWidget, QHBoxLayout, QVBoxLayout, QProgressBar, QPushButton, QGridLayout, QLabel, QLineEdit, QSizePolicy
from PyQt5.QtWidgets import QMenuBar, QMenu, QFileDialog, QFrame, QTabWidget, QPlainTextEdit, QComboBox, QCheckBox, QShortcut, QTextEdit, QMessageBox, QDialog, QDialogButtonBox, QSpinBox
from pathlib import Path
from skimage.filters import threshold_otsu, threshold_minimum

from PIL import Image
import multiprocessing as mp
import mrcfile
import qimage2ndarray as q2n
import numpy as np
import psutil
from time import sleep
# import carbon_edge_detector as ced
from cryosparc.tools import Dataset
import json
import grid_edge_detector.carbon_edge_detector as ced
# import ced
from skimage.draw import disk
import toml
# from carbon_edge_detector
# import .carbon_edge_detector as ced
from datetime import datetime
from collections import defaultdict
import starfile
from traceback import format_exception
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray


TIMER = defaultdict(lambda:0)
Index = namedtuple("Index", ["row", "column"])
CURRENTCOUNTER = 0
NUMBER_OF_COLUMNS = 7
CELL_PADDING = 5 # all sides
MAX_CORES = psutil.cpu_count()
CURRENTLY_RUNNING = False
DISABLE_FUNCTION = None
ALLOWED_SUFFIXES = set([".mrc", ".rec", ".MRC", ".REC", ".png", ".jpg", ".jpeg"])
CONFIG_DIR = Path().home() / ".config" / "GridEdgeDetector"
if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir(parents=True)

CONFIG_FILE = CONFIG_DIR / "config.toml"
DEFAULT_CONFIG = {
    "title":"Grid edge detector configs", 
    "command_line_arguments":{
      "cryosparc":{
          "run":False,
          "job_path":""
      },
      "relion":{
          "run":False,
          "picking_path":""
      },
      "mask":{
          "save":False,
          "output_dir":str(Path().home())
      },
      "masked":{
          "save":False,
          "output_dir":str(Path().home()),
          "type":"mean"
      },
      "input_files":"",
      "pixel_size":1,
      "recursive":False
      
    },
    "parameters":{"threshold":0.005, "gridsizes":[1.2], "njobs":1, "to_size":100, "outside_circle_coverage":0.05, "inner_circle_coverage":0.2, "detect_ring":True, "ring_width":0.02, "wobble":0,"edge_distance":0,
                   "high_pass_filter":0, "crop":100, "return_ring_width":0},
    # "parameters" :{"threshold":0.02, "gridsizes":[2.0], "njobs":1, "resize_value":7.0, "resize_bool":True}, 
    "files":{"filedir":str(Path().home()), "mask_file_suffix":"_mask","masked_image_file_suffix":"_masked", "cryosparc_projects_dir":"", "mask_file_type":".mrc", "masked_file_type":".mrc"},
    "misc":{"colorblind_mode":False}}

CURRENT_CONFIG = None
COLORS_DEFAULT = {"not yet":"red", "nothing found":"yellow", "mask found":"green"}
COLORS = {}
COLORS_ALTERNATIVE = {"not yet":"red", "nothing found":"orange", "mask found":"light blue"}




class mask_file:
    def __init__(self, shape, center, gridsize, distance, ring_width, inverse, path, pixelsize, edge_found, has_ice) -> None:
        self.shape = shape
        self.center = center
        self.gridsize = gridsize
        self.distance = distance
        self.ring_width = ring_width
        self.inverse = inverse
        self.path = path
        self.pixelsize = pixelsize
        self.edge_found = edge_found
        self.has_ice = has_ice
    
    def create_mask(self):
        
        if not self.edge_found:
            if self.has_ice:
                return np.ones(self.shape)
            return np.zeros(self.shape)
        if self.ring_width > 0:
            mask = np.ones(self.shape)
            orig_y, orig_x = disk(self.center, (self.gridsize + self.ring_width/2)/self.pixelsize/2, shape=self.shape)
            mask[orig_y, orig_x] = 0
            orig_y, orig_x = disk(self.center, (self.gridsize - self.ring_width/2)/self.pixelsize/2, shape=self.shape)
            mask[orig_y, orig_x] = 1
        else:
            mask = np.zeros(self.shape)
            yy,xx = disk(self.center, self.gridsize/self.pixelsize // 2, shape=self.shape,)
                
            mask[yy,xx] = 1
            if np.abs(self.distance) > 0 and np.abs(self.distance) < self.gridsize / 2:
                yy, xx = disk(self.center, np.abs(self.distance) / self.pixelsize , shape=self.shape)
                if self.distance > 0:
                    mask[yy,xx] = 0
                else:
                    mask = np.zeros_like(mask)
                    mask[yy, xx] = 1
            
        
        if self.inverse:
            mask = (mask == 0) * 1
        
        return mask

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            mf = pickle.load(f)
        return mf



def checkCrypSparcProject(path):
    project_manifest = path / "job_manifest.json"
    project_json = path / "project.json"
    workspaces_json = path / "workspaces.json"
    return project_manifest.exists() and project_json.exists() and workspaces_json.exists()

def findPickingJobs(project):
    global TIMER

    now = datetime.now()
    with open(project / "job_manifest.json", "r") as f:
        read = json.load(f)
    TIMER["read_json"] += (datetime.now() - now).total_seconds()
    now = datetime.now()
    jobs = read["jobs"]
    picking_jobs = []
    TIMER["rest_find_picking"] += (datetime.now() - now).total_seconds()
    now = datetime.now()
    for job in jobs:
        picked_particles_path = project / job / "picked_particles.cs"
        picked_micrograph_path = project / job / "picked_micrographs.cs"
        if picked_micrograph_path.exists() and picked_particles_path.exists():
            picking_jobs.append(job)
        TIMER["exists"] += (datetime.now() - now).total_seconds()
        now = datetime.now()
        
        TIMER["exists_os"] += (datetime.now() - now).total_seconds()
        now = datetime.now()
        
    return picking_jobs

class NonMrcFilesPixelSizeWidget(QDialog):
    def __init__(self, parent=None, count=None) -> None:
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        if count is None:
            text = "There are files without given pixel size.\nPlease confirm the pixel size for these files."
        else:
            text = f"There are {count} files without given pixel size.\nPlease confirm the pixel size for these files."
        label = QLabel(text)
        pixelsizelabel = QLabel("Pixelsize [Å]")
        self.pixelsizelineedit = QLineEdit("1.0")
        self.pixelsizelineedit.setValidator(CorrectDoubleValidator(0.01, None, 1.0))
        lowerLayout = QHBoxLayout()
        lowerLayout.addWidget(pixelsizelabel)
        lowerLayout.addWidget(self.pixelsizelineedit)
        self.layout().addWidget(label)
        self.layout().addLayout(lowerLayout)

        self.buttonLayout = QHBoxLayout()
        self.confirmButton = QPushButton("Confirm")
        self.confirmButton.clicked.connect(self.confirm)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.cancel)
        self.buttonLayout.addWidget(self.confirmButton)
        self.buttonLayout.addWidget(self.cancelButton)
        
        self.layout().addLayout(self.buttonLayout)
        
    
    def cancel(self):
        self.reject()

    def confirm(self):
        self.accept()

    def getPixelSize(self):
        return float(self.pixelsizelineedit.text())


def applyMask(analyserPath, mask, index):
    from cryovia.cryovia_analysis.analyser import AnalyserWrapper, Analyser
    analyser = Analyser.load(analyserPath, index=index)
    analyser.applyMask(mask)
    analyser.save()

def create_default_config_file(overwrite=False):
    if not overwrite:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "r") as f:
                config = toml.load(f)
            if not check_config(config):
                raise ValueError
        except Exception as e:
            overwrite = True
    
    
    
    if not overwrite and CONFIG_FILE.exists():
        return
    
    with open(CONFIG_FILE, "w") as f:
        a = toml.dump(DEFAULT_CONFIG,f)


def check_config(config):
    def check_dic(first_dic, second_dic):
        for key, value in first_dic.items():
            if key not in second_dic:
                return False
            if type(value) != type(second_dic[key]):
                return False
            if isinstance(second_dic[key], dict):
                if not check_dic(value, second_dic[key]):
                    return False
            
        return True
    global DEFAULT_CONFIG
    return_value = check_dic(DEFAULT_CONFIG, config)
    return return_value 

def load_config_file():
    try:
        with open(CONFIG_FILE, "r") as f:
            configs = toml.load(f)
        if not check_config(configs):
            raise ValueError
    except Exception as e:
        
        create_default_config_file(True)
        with open(CONFIG_FILE, "r") as f:
            configs = toml.load(f)
    return configs


def inverseMask(mask):
    return (mask == 0) * 1

def reevaluateMask(center, gridsize, ps, shape, distance, ring_width, inverse):
    
    

    mask = np.zeros(shape, dtype=np.uint8)
    if ring_width > 0:
        mask = np.ones(shape, dtype=np.uint8)
        orig_y, orig_x = disk(center, (gridsize + ring_width/2)/ps/2, shape=shape)
        mask[orig_y, orig_x] = 0
        orig_y, orig_x = disk(center, (gridsize - ring_width/2)/ps/2, shape=shape)
        mask[orig_y, orig_x] = 1
    else:
        mask = np.zeros(shape, dtype=np.uint8)
        yy,xx = disk(center, gridsize/ps // 2, shape=shape,)
        mask[yy,xx] = 1
        if np.abs(distance) > 0 and np.abs(distance) < gridsize / 2:
            yy, xx = disk(center, np.abs(distance) / ps , shape=shape)
            if distance > 0:
                mask[yy,xx] = 0
            else:
                mask = np.zeros_like(mask)
                mask[yy, xx] = 1
    if inverse:
        mask = (mask==0)* 1
    return mask

def reevaluateMaskMP(center, gridsize, ps, shape, threshold, max_, distance, ring_width, inverse):
    if max_ > threshold:
        return reevaluateMask(center, gridsize, ps, shape, distance, ring_width, inverse), True
    else:
        return np.ones(shape, np.uint8), False

def run_parallel(idx, fn, metadata, parameters):
    try:
        # print(f"Running parallel {idx}")
        mask, hist_data, gridsize, thresholdImage, usedImage = ced.find_grid_hole_per_file(fn, parameters["to_size"], [i * 10000 for i in metadata["Gridsize"]], metadata["Threshold"],return_hist_data=True,
                                                                coverage_percentage=metadata["inside_coverage"],outside_coverage_percentage=metadata["outside_coverage"],
                                                                  ring_width=parameters["ring_width"]*10000,detect_ring=parameters["detect_ring"], pixel_size=metadata["Pixel spacing"], wobble=parameters["wobble"],
                                                                  high_pass=metadata["high_pass_filter"], crop=parameters["crop"], distance=metadata["distance"], return_ring_width=metadata["return_ring_width"])
        # mask, hist_data, gridsize = ced.mask_carbon_edge_per_file(fn, [i * 10000 for i in metadata["Gridsize"]], metadata["Threshold"], metadata["Pixel spacing"], get_hist_data=True,to_resize=to_resize, resize=resize_value)
        resized_mask = Image.fromarray(mask)
        resized_mask.thumbnail((200,200))
        found_edge = len(np.unique(mask)) > 1
        return idx, mask,thresholdImage,usedImage,  hist_data, gridsize, np.array(resized_mask), found_edge
    except Exception as e:
        e = traceback.format_exc()
        print(e)
        return tuple([e])


class Worker(QObject):

    finished = pyqtSignal()
    progress = pyqtSignal(tuple)
    def __init__(self, indexes, image_datas,parameters):
        super().__init__()
        
        self.indexes = indexes
        self.image_datas = image_datas
        self.parameters = parameters
        # self.threshold = threshold

    def run_old(self):
        def callback(result):
            self.progress.emit(result)
            
        now = datetime.now()
        with mp.get_context("spawn").Pool(self.parameters["njobs"]) as pool:
            now = datetime.now()
            result = [pool.apply_async(run_parallel, [idx, data.fn, data.metadata, self.parameters ], callback=callback) for idx, data in zip(self.indexes, self.image_datas)]
            [res.get() for res in result]

        pool.join()
        
    def run(self):
        def callback(result):
            self.progress.emit(result)
        # ray.init(num_cpus=self.parameters["njobs"])
        
        remote_run_parallel = ray.remote(num_cpus=max(1,np.ceil(ray.cluster_resources()["CPU"]/self.parameters["njobs"])))(run_parallel)
        # remote_run_parallel = ray.remote(run_parallel)
        
        result = [remote_run_parallel.remote(idx, data.fn, data.metadata, self.parameters) for idx, data in zip(self.indexes, self.image_datas)]
        

        ready_ids, remaining_ids = ray.wait(result)
        # Get the results of completed tasks
        results = ray.get(ready_ids)

        # Process the results as they become available
        for result in results:
            # result = ray.get(result)
            callback(result)

        # Continue processing remaining tasks, if needed
        while remaining_ids:
            ready_ids, remaining_ids = ray.wait(remaining_ids)
            results = ray.get(ready_ids)
            for result in results:
                # result = ray.get(result)
                callback(result)


        # for res in result:
        #     res = ray.get(res)
        #     callback(res)
        # ray.shutdown()
        


        self.finished.emit()
    




def loadImageData(fn, dataset, config, cryoSparcInfo=None):
    return image_data(fn, dataset, True, config, cryosparc_info=cryoSparcInfo)



class image_data:
    def __init__(self, fn, dataset, for_mp=False, config=None, cryosparc_info=None, rlnInfo=None) -> None:
        global CURRENT_CONFIG
        self.metadata = {"inverse":False}
        self.fn = Path(fn)
        self.cryosparc_info = cryosparc_info
        self.rlnInfo = rlnInfo
        self.mean_ = None
        self.median_ = None
        if dataset is None:
            self.dataset = None
        else:
            self.dataset = dataset.name
        if self.fn.suffix in [".mrc", ".MRC", ".rec", ".REC"]:
            with mrcfile.open(self.fn,permissive=True) as f:
                data = f.data * 1
                self.mean_ = np.mean(data)
                self.median_ = np.median(data)
                self.metadata["Dimensions"] = data.shape
                middle = np.median(data)
                std = np.std(data)
                left = middle - std * 2
                right = middle + std * 2
                data = np.clip(data, left, right)
                data = Image.fromarray(data)
                data.thumbnail((200,200))
                data = np.array(data)
                
                self.metadata["Pixel spacing"] = float(f.voxel_size["x"])
                
                

        else:
            img = Image.open(self.fn).convert("L")
            self.metadata["Dimensions"] = img.size[::-1]
            img.thumbnail((200,200))
            data = np.array(img)
            self.mean_ = np.mean(data)
            self.median_ = np.median(data)
            self.metadata["Pixel spacing"] = 1
            middle = np.median(data)
            std = np.std(data)
            left = middle - std * 2
            right = middle + std * 2
            data = np.clip(data, left, right)
        if config is None:
            self.metadata["Gridsize"] = CURRENT_CONFIG["parameters"]["gridsizes"]
            self.metadata["Threshold"] = CURRENT_CONFIG["parameters"]["threshold"]
            self.metadata["distance"] = CURRENT_CONFIG["parameters"]["edge_distance"]
            self.metadata["return_ring_width"] = CURRENT_CONFIG["parameters"]["return_ring_width"]
            self.metadata["outside_coverage"] = CURRENT_CONFIG["parameters"]["outside_circle_coverage"]
            self.metadata["inside_coverage"] = CURRENT_CONFIG["parameters"]["inner_circle_coverage"]
            self.metadata["high_pass_filter"] = CURRENT_CONFIG["parameters"]["high_pass_filter"]
        else:
            self.metadata["Gridsize"] = config["parameters"]["gridsizes"]
            self.metadata["Threshold"] = config["parameters"]["threshold"]
            self.metadata["distance"] = config["parameters"]["edge_distance"]
            self.metadata["return_ring_width"] = config["parameters"]["return_ring_width"]
            self.metadata["outside_coverage"] = config["parameters"]["outside_circle_coverage"]
            self.metadata["inside_coverage"] = config["parameters"]["inner_circle_coverage"]
            self.metadata["high_pass_filter"] = config["parameters"]["high_pass_filter"]

        if for_mp:
            self.image_ = data
        else:

            self.image_  = q2n.gray2qimage(data, True)
        self.mask_ = None
        self.original_mask_ = None
        self.hist_data_ = None
        self.best_gridsize_ = None
        self.changed = True
        self.found_edge = False
        self.thresholdImage = None
        self.usedImage = None


    @property
    def mean(self):

        if self.mean_ is None:
            self.mean_ = np.mean(self.original_image)
        return self.mean_
    
    @property
    def median(self):
        if self.median_ is None:
            self.median_ = np.median(self.original_image)
        return self.median_


    @property
    def percentage(self):
        if self.original_mask is None:
            return np.nan
        else:
            return np.sum(self.original_mask == 0) / self.original_mask.size
        



    @property
    def original_image(self):
        if self.fn.suffix in [".mrc", ".MRC", ".rec", ".REC"]:
            with mrcfile.open(self.fn,permissive=True) as f:
                data = f.data * 1   
        else:
            data = np.array(Image.open(self.fn).convert("L"))
        return data
    
    def convert(self):
        self.image_ = q2n.gray2qimage(self.image_, True)

    @property
    def title(self):
        return self.fn.name
        

    @property
    def image(self):
        return self.image_
    
    @image.setter
    def image(self, value):
        self.changed = True
        self.image_ = value
    




    @property
    def mask(self):
        return self.mask_
    
    @mask.setter
    def mask(self, value):
        self.changed = True
        self.mask_ = value
    
    @property
    def original_mask(self):
        return self.original_mask_
    
    @original_mask.setter
    def original_mask(self, value):
        self.mask = q2n.array2qimage(value, True)
        self.changed = True
        
        self.original_mask_ = value

    @property
    def hist_data(self):
        return self.hist_data_

    @hist_data.setter
    def hist_data(self, value):
        self.changed = True
        self.hist_data_ = value

    @property
    def best_gridsize(self):
        return self.best_gridsize_
    
    @best_gridsize.setter
    def best_gridsize(self, value):
        self.changed = True
        self.best_gridsize_ = value
    

class PreviewDelegate(QStyledItemDelegate):

    def __init__(self, parent=None) -> None:
        self.shape = np.array([320,160])
        super().__init__(parent)
    def paint(self, painter, option, index):
        global COLORS
        
        # data is our preview object
        
        data = index.model().data(index, Qt.DisplayRole)
        if data is None:
            return
        # painter.save()
        # width = option.rect.width() - CELL_PADDING * 2
        width = (option.rect.width() - CELL_PADDING * 2) // 2
        height = option.rect.height() - CELL_PADDING * 2

        # option.rect holds the area we are painting on the widget (our table cell)
        # scale our pixmap to fit
        scaled = data.image.scaled(
            width,
            height,
            aspectRatioMode=Qt.KeepAspectRatio,
            transformMode=Qt.TransformationMode.SmoothTransformation
        )
        # Position in the middle of the area.
        x = int(CELL_PADDING + (width - scaled.width()) / 2)
        y = int(CELL_PADDING + (height - scaled.height()) / 2)


        if data.mask is None:
            color = QColor(COLORS["not yet"])
            
        elif data.original_mask is not None and data.found_edge:
            color = QColor(COLORS["mask found"])
        else:
            color = QColor(COLORS["nothing found"])
        painter.drawImage(option.rect.x() + x, option.rect.y() + y, scaled)
        painter.setPen(QPen(color, 3))
        painter.drawRect(option.rect.x() + x - 2, option.rect.y() + y - 2, scaled.width() * 2 + 4, scaled.height() + 4 )
        if data.mask is not None:
            if data.found_edge:
                scaled_mask = data.mask.scaled(width, height, aspectRatioMode=Qt.KeepAspectRatio)
            else:
                scaled :QImage
                scaled_mask = scaled.copy()
                if np.max(data.original_mask) == 1:
                    scaled_mask.fill(QColor("white"))
                else:
                    scaled_mask.fill(QColor("black"))
            # scaled_mask.fill(QColor("white"))
            painter.drawImage(option.rect.x() + x + scaled.width(), option.rect.y() + y, scaled_mask)
        
        # painter.restore()
    def sizeHint(self, option, index):
        # All items the same size.
        return QSize(self.shape[0], self.shape[1])


class customSelectionModel(QItemSelectionModel):
    def __init__(self, model):
        
        super().__init__()
        self.setModel(model)

    def selectedIndexes(self):
         
        idxs = super().selectedIndexes()
        new_idxs = [idx.row() * self.model().columnCount() + idx.column() for idx in idxs]
        idxs = [idx for idx, new_idx in zip(idxs, new_idxs) if new_idx < len(self.model().previews)]
        return idxs

class PreviewModel(QAbstractTableModel):
    def __init__(self, parent):
        super().__init__(parent)
        
        # .data holds our data for display, as a list of Preview objects.
        self.previews = []

    def data(self, index, role):
        try:
            data = self.previews[index.row() * self.columnCount() + index.column() ]
            
        except IndexError:

            return

        if role == Qt.DisplayRole:
            
            return data   # Pass the data to our delegate to draw.

        if role == Qt.ToolTipRole:
            return data.title

    def deleteIdxs(self, idxs):
        self.layoutAboutToBeChanged.emit()
        idxs = sorted(idxs, reverse=True)
        for idx in idxs:
            self.previews.pop(idx)
        self.layoutChanged.emit()


    def columnCount(self, index=None):
        return max(1,self.parent().size().width() // self.parent().delegate.shape[0])

    def rowCount(self, index=None):
        n_items = len(self.previews)
        return math.ceil(n_items / self.columnCount())



class ImageViewer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.view = QTableView()
        
        self.setMinimumSize(500,500)
        self.view.horizontalHeader().hide()
        self.view.verticalHeader().hide()
        self.view.setGridStyle(Qt.NoPen)

        self.delegate = PreviewDelegate()
        self.view.setItemDelegate(self.delegate)
        self.model = PreviewModel(self)
        self.view.setModel(self.model)
        
        self.view.setSelectionModel(customSelectionModel(self.model))
        self.view.selectionModel().selectionChanged.connect(self.selectionChanged)

        # palette = QPalette()
        # palette.setColor(QPalette.Highlight, QColor("red"))

        # self.view.setPalette(palette)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.view)
        # self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed))

        self.setAcceptDrops(True)
        
        self.scroll_timer = QTimer()
        self.scroll_timer.setInterval(500)  # Adjust the interval as needed
        self.scroll_timer.setSingleShot(True)
        self.scroll_timer.timeout.connect(self.on_scroll_stopped)
        self.scrollCounter = 0



    def changeImageSize(self, direction):
        self.scrollCounter += direction
        new_shape = self.delegate.shape + np.array([10,5]) * self.scrollCounter
        if new_shape[0] < 40 or new_shape[1] <20:
            new_shape = np.array([40,20])
        elif new_shape[0] > 1280 or new_shape[1] >640:
            new_shape = np.array([1280,640])
        # self.shapeLabel.setText(f"[{new_shape[0]}, {new_shape[1]}]")
        self.scroll_timer.start()

    def on_scroll_stopped(self):
        if self.scrollCounter != 0:
            self.delegate.shape += np.array([10,5]) * self.scrollCounter
            if self.delegate.shape[0] < 40 or self.delegate.shape[1] <20:
                self.delegate.shape = np.array([40,20])
            elif self.delegate.shape[0] > 1280 or self.delegate.shape[1] >640:
                self.delegate.shape = np.array([1280,640])
            self.model.layoutChanged.emit()
            self.view.resizeColumnsToContents()
            self.view.resizeRowsToContents()
            self.scrollCounter = 0
    def wheelEvent(self, a0: QWheelEvent) -> None:
        modifiers = QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            if a0.angleDelta().y() > 0:
                self.scrollCounter -= 1
            elif a0.angleDelta().y() < 0:
                self.scrollCounter += 1
            new_shape = self.delegate.shape + np.array([10,5]) * self.scrollCounter
            if new_shape[0] < 40 or new_shape[1] <20:
                new_shape = np.array([40,20])
            elif new_shape[0] > 1280 or new_shape[1] >640:
                new_shape = np.array([1280,640])
            # self.shapeLabel.setText(f"[{new_shape[0]}, {new_shape[1]}]")
            self.scroll_timer.start()
            a0.accept()
            return
        return super().wheelEvent(a0)


    def dragEnterEvent(self, event ) -> None:
        global ALLOWED_SUFFIXES
        if event.mimeData().hasUrls():
            files = [Path(u.toLocalFile()) for u in event.mimeData().urls()]
            if any([file.suffix not in ALLOWED_SUFFIXES for file in files]):
                event.ignore()
                return
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event) -> None:
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.load_files(files)
                


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            global CURRENTLY_RUNNING
            if CURRENTLY_RUNNING:
                return

            idxs = self.view.selectionModel().selectedIndexes()
            new_idxs = [idx.row() * self.model.columnCount() + idx.column() for idx in idxs]
            self.model.deleteIdxs(new_idxs)
            self.view.selectionModel().clearSelection()
            # row = self.currentRow()
            # self.removeRow(row)
        else:
            super().keyPressEvent(event)

        # Add a bunch of images.
    def sizeHint(self):
        # All items the same size.
        return QSize(1250, 850)
        
    def selectionChanged(self, event=None):
        idxs = self.view.selectionModel().selectedIndexes()
        self.parent().metadataWidget.show_data(idxs)


    def load_files(self, files, dataset=None, njobs=1, pixelSizes=None, cryoSparcInfo=None, rlnInfo=None):
        # global CURRENTCOUNTER
        # def getCallback(maxNumber):
        #     def callback(result):
        #         global CURRENTCOUNTER
        #         CURRENTCOUNTER += 1
        #         self.mainwidget.setProgress(CURRENTCOUNTER/maxNumber * 100, True)
        #     return callback
        a = self.parent().metadataWidget.runWidget

        njobs = int(a.configLineEdits["njobs"][0].text())

        if pixelSizes is None:
            pixelSizes = {path:None for path in files}
        if cryoSparcInfo is None:
            cryoSparcInfo = {path:None for path in files}
        if rlnInfo is None:
            rlnInfo = {path:None for path in files}
        pixelSizes = {Path(path):ps for path, ps in pixelSizes.items()}
        nonMrcFiles = [path for path in files if Path(path).suffix != ".mrc"]
        if len(nonMrcFiles) > 0:
            pixelSizeWidget = NonMrcFilesPixelSizeWidget(self, len(nonMrcFiles))
            result = pixelSizeWidget.exec()
            if result == 0:
                return
            ps = pixelSizeWidget.getPixelSize()
            for file in nonMrcFiles:
                pixelSizes[file] = ps 


        if njobs <= 1:
            for n, fn in enumerate(files):

                item = image_data(fn, dataset, cryosparc_info=cryoSparcInfo[fn], rlnInfo=rlnInfo[fn])
                if pixelSizes[fn] is not None:
                    item.metadata["Pixel spacing"] = pixelSizes[fn] 
                self.model.previews.append(item)
                self.parent().setProgress((1+ n)/len(files) * 100)
        else:
            # CURRENTCOUNTER = 0
            global CURRENT_CONFIG
            loadImageData_remote = ray.remote(num_cpus=max(1, np.ceil(ray.cluster_resources()["CPU"]/njobs)))(loadImageData)
            result = [loadImageData_remote.remote(file, dataset, CURRENT_CONFIG,cryoSparcInfo[file]) for file in files]

            ready_ids, remaining_ids = ray.wait(result)
            results = ray.get(ready_ids)
            n = 0
            for item in results:
                # result = ray.get(result)
                item.convert()
                if pixelSizes[item.fn] is not None:
                    item.metadata["Pixel spacing"] = pixelSizes[item.fn] 
                self.model.previews.append(item)
                self.parent().setProgress((1+ n)/len(files) * 100)
                n += 1

            # Continue processing remaining tasks, if needed
            while remaining_ids:
                ready_ids, remaining_ids = ray.wait(remaining_ids)
                results = ray.get(ready_ids)
                for item in results:
                    # result = ray.get(result)
                    item.convert()
                    if pixelSizes[item.fn] is not None:
                        item.metadata["Pixel spacing"] = pixelSizes[item.fn] 
                    self.model.previews.append(item)
                    self.parent().setProgress((1+ n)/len(files) * 100)
                    n += 1

            # for n,(res, fn) in enumerate(zip(result, files)):
            #     item:image_data = ray.get(res)
            #     item.convert()
            #     if pixelSizes[fn] is not None:
            #         item.metadata["Pixel spacing"] = pixelSizes[fn] 
            #     self.model.previews.append(item)
            #     self.parent().setProgress((1+ n)/len(files) * 100)



            # with mp.get_context("spawn").Pool(njobs) as pool:
            #     result = [pool.apply_async(loadImageData, [file, dataset, CURRENT_CONFIG,cryoSparcInfo[file]]) for file in files]
            #     for n,(res, fn) in enumerate(zip(result, files)):
            #         item:image_data = res.get()
            #         item.convert()
            #         if pixelSizes[fn] is not None:
            #             item.metadata["Pixel spacing"] = pixelSizes[fn] 
            #         self.model.previews.append(item)
            #         self.parent().setProgress((1+ n)/len(files) * 100)
                    
        self.model.layoutChanged.emit()

        self.view.resizeRowsToContents()
        self.view.resizeColumnsToContents()



class carbonIceDecider(QWidget):


    def __init__(self, parent, means) -> None:
        super().__init__()
        self.customParent = parent
        self.setLayout(QVBoxLayout())
        self.carbonLabel = QLabel("Carbon", alignment=Qt.AlignLeft)
        self.iceLabel = QLabel("Ice", alignment=Qt.AlignRight)
        self.labelLayout = QHBoxLayout()
        self.labelLayout.addWidget(self.carbonLabel)
        self.labelLayout.addWidget(self.iceLabel)
        self.line = None
        self.edges = None
        self.values = None

        self.thresholdText = QLineEdit(self)
        self.thresholdText.editingFinished.connect(self.updateLine)
        self.thresholdText.setValidator(QDoubleValidator())
        self.thresholdLabel = QLabel("Threshold")
        self.thresholdPredictButton = QPushButton("Predict Threshold")
        self.thresholdPredictButton.clicked.connect(self.predictThreshold)

        self.thresholdLayout = QHBoxLayout()
        self.thresholdLayout.addWidget(self.thresholdLabel, alignment=Qt.AlignLeft)
        self.thresholdLayout.addWidget(self.thresholdText, alignment=Qt.AlignLeft)
        self.thresholdLayout.addWidget(self.thresholdPredictButton, alignment=Qt.AlignLeft)

        self.applyLayout = QHBoxLayout()
        self.applyButton = QPushButton("Apply")
        self.okButton = QPushButton("OK")
        self.cancelButton = QPushButton("Close")
        self.applyButton.clicked.connect(lambda :self.apply("apply"))
        self.okButton.clicked.connect(lambda : self.apply("ok"))
        self.cancelButton.clicked.connect(lambda :self.apply("close"))

        self.applyLayout.addWidget(self.applyButton)
        
        self.applyLayout.addWidget(self.okButton)
        self.applyLayout.addWidget(self.cancelButton)


        self.plotWidget = PlotWidget(self, "white")
        self.plotWidget.setMouseEnabled(False, False)

        self.layout().addLayout(self.labelLayout)
        self.layout().addWidget(self.plotWidget)
        self.layout().addLayout(self.thresholdLayout)
        self.layout().addLayout(self.applyLayout)


        
        self.updatePlot(means)
        

    def apply(self, s):
        
        
        if s == "ok" or s == "apply":
            self.customParent.applyThreshold(self.line.getPos()[0])
        if s == "close" or s == "ok":
            self.close()
            

    
    def updatePlot(self, means):

        hist = self.plotWidget
        hist.clear()

        

        values, edges = np.histogram(means, 20)
        self.values = values
        self.edges = edges



        self.bargraph = pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=values)
        
        self.line = pg.InfiniteLine(0,pen="red", movable=True)
        self.plotWidget.addItem(self.bargraph)
        self.plotWidget.addItem(self.line)
        self.line.sigPositionChangeFinished.connect(self.updateThresholdLabel)
        self.predictThreshold()
        


    def setXrange(self, threshold):
        self.plotWidget.setXRange(np.min([np.min(self.edges), threshold]), np.max([np.max(self.edges), threshold]))

    def updateThresholdLabel(self):
        newThreshold = self.line.getPos()[0]
        self.thresholdText.setText(str(newThreshold))
        self.setXrange(newThreshold)

    def updateLine(self, newThreshold=None):
        if newThreshold is None:
            newThreshold = float(self.thresholdText.text())
        self.line.setPos(newThreshold)
        self.setXrange(newThreshold)
    
    def predictThreshold(self):

        middle_edges = [np.mean([self.edges[i], self.edges[i+1]]) for i in range(len(self.edges)-1)]
        try:
            newThreshold = threshold_minimum(hist=(self.values, middle_edges))
        except Exception as e:

            newThreshold = np.median(self.edges)
        self.updateLine(newThreshold)
        self.updateThresholdLabel()
    

    def sizeHint(self):
        # All items the same size.
        return QSize(600, 500)







class histWidget(QFrame):


    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Panel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.setLayout(QHBoxLayout())
        self.plotTabWidget = QTabWidget(self)

        self.plotTabWidget.addTab(PlotWidget(self, "white"), "hist")
        
        # self.hist.getPlotItem().setLogMode(False, True)
        self.layout().addWidget(self.plotTabWidget)
        self.plotItems = {}
    
    def updatePlot(self, id, idx):
        id:image_data
        tab_count = self.plotTabWidget.count()
        for idx in range(tab_count-1, 0, -1):
            self.plotTabWidget.removeTab(idx)
        hist = self.plotTabWidget.widget(0)
        hist.clear()
        self.plotItems = {}
        if id.hist_data is None:
            self.plotTabWidget.setTabText(0, "Hist")
            return
        self.plotTabWidget.removeTab(0)
        for key in sorted(id.hist_data.keys()):
            current_plot_widget = PlotWidget(self.plotTabWidget, "white")
            

            hist_data = id.hist_data[key]
            tab_name = f"{round(key / 10000, 3)} µm"
        # hist_data = id.hist_data[id.best_gridsize]
            edges = hist_data["edges"]
            values = hist_data["values"]
            threshold = id.metadata["Threshold"]

            bargraph = pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=np.log(values),)
            
            line = pg.InfiniteLine(threshold,pen="red", movable=True)
            # line.sigPositionChanged.connect(self.updateThreshold)
            line.sigPositionChangeFinished.connect(self.updateMask)
            current_plot_widget.setXRange(np.min(edges), max(np.max(edges), threshold))
            current_plot_widget.addItem(bargraph)
            current_plot_widget.addItem(line)
            # ax.bar(edges[:-1], values, width=np.diff(edges))

            self.plotItems[tab_name] = {"line":line, "bargraph":bargraph, "imagedata":id, "gridsize":key, "idx":idx }
            self.plotTabWidget.addTab(current_plot_widget,tab_name)

    def clearAll(self):
        tab_count = self.plotTabWidget.count()
        for idx in range(tab_count-1, 0, -1):
            self.plotTabWidget.removeTab(idx)
        hist = self.plotTabWidget.widget(0)
        hist.clear()
        self.plotTabWidget.setTabText(0, "Hist")
        self.plotItems = {}

    # def updateThreshold(self, line=None):

    #     line: pg.InfiniteLine

    #     self.parent().runWidget.thresholdLineEdit.setText(str(line.getPos()[0]))
    #     self.parent().parent().imageviewer.
        
    
    def updateMask(self, line=None, newthreshold=None):
        if line is not None:
            newthreshold = line.getPos()[0]
        
        for key, value in self.plotItems.items():
            value["line"].setPos(newthreshold)
            
        pass
        if len(self.plotItems.keys()) != 0:
            # newthreshold = line.getPos()[0]
            
            name = self.plotTabWidget.tabText(self.plotTabWidget.currentIndex())
            id = self.plotItems[name]["imagedata"]
            hist_data = id.hist_data[self.plotItems[name]["gridsize"]]
            edges = hist_data["edges"]
            values = hist_data["values"]
            self.plotTabWidget.currentWidget().setXRange(np.min(edges), max(np.max(edges), newthreshold))
            if np.max(edges) >= newthreshold:
                gridsize = self.plotItems[name]["gridsize"]
                new_mask = reevaluateMask(hist_data["center"], gridsize, id.metadata["Pixel spacing"],id.metadata["Dimensions"], id.metadata["distance"], id.metadata["return_ring_width"], id.metadata["inverse"])

                
                # id.hist_data = hist_data
                
                
            else:
                for idx in range(self.plotTabWidget.count()):
                    widget = self.plotTabWidget.widget(idx)
                    
                    name = self.plotTabWidget.tabText(idx)
                    id = self.plotItems[name]["imagedata"]
                    hist_data = id.hist_data[self.plotItems[name]["gridsize"]]
                    edges = hist_data["edges"]
                    values = hist_data["values"]
                    self.plotTabWidget.currentWidget().setXRange(np.min(edges), max(np.max(edges), newthreshold))
                    if np.max(edges) >= newthreshold:
                        gridsize = self.plotItems[name]["gridsize"]
                        new_mask = reevaluateMask(hist_data["center"], gridsize, id.metadata["Pixel spacing"],id.metadata["Dimensions"], id.metadata["distance"], id.metadata["return_ring_width"], id.metadata["inverse"])
                        break
                else:
                    gridsize = self.plotItems[name]["gridsize"]
                    new_mask = np.ones(id.metadata["Dimensions"], np.uint8)
            
            id.mask = q2n.array2qimage(new_mask, True)
            id.original_mask = new_mask
            id.found_edge = len(np.unique(new_mask)) > 1
            id.best_gridsize = gridsize
            id.metadata["Threshold"] = newthreshold
            
            self.parent().thumbnailWidget.load_images(id)
            
            for idx in range(self.plotTabWidget.count()):
                widget = self.plotTabWidget.widget(idx)
                
                name = self.plotTabWidget.tabText(idx)
                id = self.plotItems[name]["imagedata"]
                hist_data = id.hist_data[id.best_gridsize]
                edges = hist_data["edges"]
                values = hist_data["values"]
                widget.setXRange(np.min(edges), max(np.max(edges), newthreshold))
            idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
            if len(idxs) == 1:
                self.parent().parent().imageviewer.view.model().dataChanged.emit(idxs[0], idxs[0])
                self.parent().metadataWidget.thresholdLineEdit.setText(str(newthreshold))
                
    def sizeHint(self):
        # All items the same size.
        return QSize(500, 250)
        
            
    # def sizeHint(self):
    #     return QSize(240,240)

class thumbnailWidget(QFrame):
    shape = 100


    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Panel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.imageLabel = QLabel()
        self.maskLabel = QLabel()
        self.thresholdImageLabel = QLabel()
        self.usedImageLabel = QLabel()

        self.imagePixmap = QPixmap(self.shape,self.shape)
        self.maskPixmap = QPixmap(self.shape,self.shape)
        self.thresholdImagePixmap = QPixmap(self.shape,self.shape)
        self.usedImagePixmap = QPixmap(self.shape,self.shape)

        self.imagePixmap.fill(QColor("white"))
        self.maskPixmap.fill(QColor("white"))
        self.thresholdImagePixmap.fill(QColor("white"))
        self.usedImagePixmap.fill(QColor("white"))

        self.imageLabel.setPixmap(self.imagePixmap)
        self.maskLabel.setPixmap(self.maskPixmap)
        self.thresholdImageLabel.setPixmap(self.thresholdImagePixmap)
        self.usedImageLabel.setPixmap(self.usedImagePixmap)

        self.setLayout(QGridLayout())
        self.imageLabel.setStyleSheet("border: 1px solid black")
        self.maskLabel.setStyleSheet("border: 1px solid black")
        self.thresholdImageLabel.setStyleSheet("border: 1px solid black")
        self.usedImageLabel.setStyleSheet("border: 1px solid black")

        self.placeholder = QWidget()
        
        
        self.layout().addWidget(self.imageLabel,0,0)
        # self.layout().addWidget(self.placeholder,0,1)
        self.layout().addWidget(self.maskLabel,0,3)
        self.layout().addWidget(self.usedImageLabel,0,1)
        self.layout().addWidget(self.thresholdImageLabel,0,2)
        # self.layout().setColumnStretch(1,1)

        self.usedImageLabel.setToolTip("The internal image used for edge detection after resizing, cropping, high pass filtering, clipping and normalizing.")
        self.thresholdImageLabel.setToolTip("Image of differences between mean inside the hole and outside with hole centers at every pixel. This image is larger than the original image because the hole center can be outside of the image.")
        self.imageLabel.setToolTip("Original image")
        self.maskLabel.setToolTip("Mask")

    def load_images(self, data:image_data):
        image = data.image
        scaled = image.scaled(
            self.shape,
            self.shape,
            aspectRatioMode=Qt.KeepAspectRatio,
            transformMode=Qt.TransformationMode.SmoothTransformation
        )
        self.imagePixmap = QPixmap.fromImage(scaled)
        self.imageLabel.setPixmap(self.imagePixmap)

        if data.mask is not None:
            
            image = data.mask
            scaled = image.scaled(
                self.shape,
                self.shape,
                aspectRatioMode=Qt.KeepAspectRatio,
            )
            self.maskPixmap = QPixmap.fromImage(scaled)
            self.maskLabel.setPixmap(self.maskPixmap)
        else:
            self.maskPixmap.fill(QColor("white"))
            self.maskLabel.setPixmap(self.maskPixmap)

        if data.thresholdImage is not None:
            
            image = data.thresholdImage
            scaled = image.scaled(
                self.shape,
                self.shape,
                aspectRatioMode=Qt.KeepAspectRatio,
            )
            self.thresholdImagePixmap = QPixmap.fromImage(scaled)
            self.thresholdImageLabel.setPixmap(self.thresholdImagePixmap)
        else:
            self.thresholdImagePixmap.fill(QColor("white"))
            self.thresholdImageLabel.setPixmap(self.thresholdImagePixmap)

        if data.usedImage is not None:
            
            image = data.usedImage
            scaled = image.scaled(
                self.shape,
                self.shape,
                aspectRatioMode=Qt.KeepAspectRatio,
            )
            self.usedImagePixmap = QPixmap.fromImage(scaled)
            self.usedImageLabel.setPixmap(self.usedImagePixmap)
        else:
            self.usedImagePixmap.fill(QColor("white"))
            self.usedImageLabel.setPixmap(self.usedImagePixmap)


    
    def clearBoth(self):
        self.imagePixmap.fill(QColor("white"))
        self.maskPixmap.fill(QColor("white"))
        self.usedImagePixmap.fill(QColor("white"))
        self.thresholdImagePixmap.fill(QColor("white"))

        self.imageLabel.setPixmap(self.imagePixmap)
        self.maskLabel.setPixmap(self.maskPixmap)
        self.usedImageLabel.setPixmap(self.usedImagePixmap)
        self.thresholdImageLabel.setPixmap(self.thresholdImagePixmap)


class CorrectDoubleValidator(QValidator):
    def __init__(self, low, top, default=None, allow_lists=False):
        super().__init__()
        
        self.low = low
        self.top = top
        self.default = default
        self.allow_lists = allow_lists
    
    def validate(self, a0: str, a1: int):
        if a0 == "" or a0 == "-":
            return QValidator.State.Intermediate, a0, a1
        try:
            if a0[0] == "-" and self.low is not None and self.low < 0 and len(a0) > 1:
                float(a0[1:])
            else:
                a2 = float(a0)
        except:
            return QValidator.State.Invalid, str(self.low), len(str(self.low))

        if self.low is not None and float(a0) < self.low:
            return QValidator.State.Intermediate, a0, a1
            a0 = self.low
        elif self.top is not None and float(a0) > self.top:
            a0 = self.top 
        return QValidator.State.Acceptable, str(a0), len(str(a0))
    
    def fixup(self, a0: str) -> str:
        if self.allow_lists:
            if len(a0.split(" ")) > 1:
                return a0
        if self.default is not None:
            return str(self.default)
        
        return str(self.low)

class CorrectIntValidator(QValidator):
    def __init__(self, low, top, default=None, allow_lists=False):
        super().__init__()
        self.low = low
        self.top = top
        self.default = default
        self.allow_lists = allow_lists
    
    def validate(self, a0: str, a1: int):
        if a0 == "" or a0=="-":
            return QValidator.State.Intermediate, a0, a1
        try:
            if a0[0] == "-" and self.low is not None and self.low < 0 and len(a0) > 1:
                int(a0[1:])
            else:
                a2 = int(a0)
        except:
            return QValidator.State.Invalid, str(self.low), len(str(self.low))

        if self.low is not None and int(a0) < self.low:
            return QValidator.State.Intermediate, a0, a1
            a0 = self.low
        elif self.top is not None and int(a0) > self.top:
            a0 = self.top 
        return QValidator.State.Acceptable, str(a0), len(str(a0))
    
    def fixup(self, a0: str) -> str:
        if self.allow_lists:
            if len(a0.split(" ")) > 1:
                return a0
        if self.default is not None:
            return str(self.default)
        return str(self.low)






class runWidget(QFrame):
    def __init__(self, parent):
        global CURRENT_CONFIG, MAX_CORES
        super().__init__(parent)
        self.setLayout(QGridLayout())
        self.deciderWindow = None

        # self.setStyleSheet("border: 1px solid black")
        # self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        # self.setLineWidth(1)
        self.setFrameShape(QFrame.Panel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        # self.njobsLabel = QLabel("# CPUs to use")
        # self.njobsLineEdit = QLineEdit(str(CURRENT_CONFIG["parameters"]["njobs"]))
        # self.thresholdLabel = QLabel("Threshold")
        # self.thresholdLineEdit = QLineEdit(str(CURRENT_CONFIG["parameters"]["threshold"]))
        # self.thresholdLineEdit.setValidator(QDoubleValidator())
        self.runAllButton = QPushButton(text="Mask all")
        self.runSelectedButton = QPushButton(text="Mask selected")
        self.iceOrCarbonButton = QPushButton("Ice or carbon")

        # self.njobsLineEdit.setValidator(CorrectIntValidator(1, MAX_CORES,))
        


        #("threshold","Threshold",float, None, None), ("gridsizes","Gridsizes", float, 0.001,None)
        # self.key_names = [("to_size","Resize to [Å/px]", float, 1,None), ("outside_circle_coverage", "Outside of circle coverage", float,0,1), ("inner_circle_coverage", "Inside of circle coverage", float, 0,1),
        #                    ("detect_ring", "Detect ring", bool,None, None), ("ring_width", "Ring width", float, 0.001, None), ("njobs", "# CPUs to use", int, 1, MAX_CORES),("wobble", "Wobble", float, 0, 0.2),
        #                    ("high_pass_filter", "High pass sigma", int, 0, None), ("crop", "Crop [Å]", int, 0, None), ]
        self.key_names = [("to_size","Resize to [Å/px]", float, 1,None), ("detect_ring", "Detect ring", bool,None, None), ("ring_width", "Ring width [µm]", float, 0.001, None), ("njobs", "# CPUs to use", int, 1, MAX_CORES),("wobble", "Wobble [%]", float, 0, 0.2),
                           ("crop", "Crop [Å]", int, 0, None), ]
        tooltips = {
            "to_size": "Resize the image to this pixel spacing before convolution for faster processing.",
            "detect_ring": "Whether to detect only the differences between the grid edge and the other values (by using a ring mask). This is useful for heavily high passed filterd images where inside and outside of the grid hole have similar contrast. If turned off all values inside of the hole will be compared to all outside values.",
            "ring_width": "Width of the ring in Å. Only used if \"detect_ring\" is active.",
            "njobs":"Number of parallel processes. Also effects loading of data.",
            "wobble":"Also check grid hole sizes which differ up to this percentage from the given grid size. Can be useful if the hole sizes are not very homogeneous.",
            "crop": "Crop the image by this amount of Å before detecting the edge. Can be used if artifacts are visible at the outer pixels or to counteract artifacts from high pass filtering."
                    }
        # tooltips = {}


        self.configLineEdits = {}

        for counter, (key, name,t, l,h) in enumerate(self.key_names):
            label = QLabel(name)
            if t is bool:
                lineedit = QCheckBox()
                lineedit.setChecked(str(CURRENT_CONFIG["parameters"][key]) == "True")
            elif t is float:
                lineedit = QLineEdit(str(CURRENT_CONFIG["parameters"][key]))
                lineedit.setValidator(CorrectDoubleValidator(l, h, CURRENT_CONFIG["parameters"][key]))
            elif t is int:
               
                lineedit = QLineEdit(str(CURRENT_CONFIG["parameters"][key]))
                lineedit.setValidator(CorrectIntValidator(l, h))
            # lineedit.setToolTip(tooltips[key])
            self.layout().addWidget(label, counter // 2, (counter % 2)*2)
            self.layout().addWidget(lineedit, counter // 2, (counter % 2)*2 + 1)
            
            self.configLineEdits[key] = (lineedit, t)
            if key in tooltips:
                label.setToolTip(tooltips[key])
                lineedit.setToolTip(tooltips[key])

        # self.toggleResizeCheckbox = QCheckBox(text="Resize")
        # self.toggleResizeCheckbox.setChecked(CURRENT_CONFIG["parameters"]["resize_bool"])
        # self.toggleResizeCheckbox.setToolTip("Resize the image during edge detection for faster calculations")
        # self.resizeLineEdit = QLineEdit(str(CURRENT_CONFIG["parameters"]["resize_value"]))
        # self.resizeLineEdit.setToolTip("Pixel spacing in px/Å for resizing")
  
        # self.resizeLineEdit.setValidator(CorrectDoubleValidator(0.001, 100))

        # self.layout().addWidget(self.toggleResizeCheckbox, 0,0)
        # self.layout().addWidget(self.resizeLineEdit,0,1)

        
        self.runAllButton.clicked.connect(self.runAll)
        self.runSelectedButton.clicked.connect(self.runSelected)
        self.iceOrCarbonButton.clicked.connect(self.iceOrCarbon)
        self.number_of_images = 1
        self.current_number_of_images = 0
        # self.layout().addWidget(self.njobsLabel, 1,0)
        # self.layout().addWidget(self.njobsLineEdit, 1,1)
    
        self.layout().addWidget(self.runAllButton,counter+1,0)
        self.layout().addWidget(self.runSelectedButton,counter+1,1)
        self.layout().addWidget(self.iceOrCarbonButton, counter +1 , 2)


    def applyThreshold(self, threshold):
        image_datas = self.parent().parent().imageviewer.model.previews
        for id_ in image_datas:
            if not id_.found_edge:
                if id_.original_mask is None:
                    id_.original_mask = np.ones(id_.metadata["Dimensions"])
                if id_.mean < threshold:
                    id_.original_mask.fill(0)
                else:
                    id_.original_mask.fill(1)
                print(id_.mean, threshold)
        self.parent().parent().imageviewer.model.layoutChanged.emit()

    def runAll(self):
        if not self.currentlyRunning:
            global DISABLE_FUNCTION
            DISABLE_FUNCTION(True)              
            rows = self.parent().parent().imageviewer.model.rowCount(None)
            columns = self.parent().parent().imageviewer.model.columnCount(None)
            count = len(self.parent().parent().imageviewer.model.previews)
            indexes = [i for i in range(count)]
            # indexes = [(row * col + col) for row in range(rows) for col in range(columns)]
            # if count % columns != 0:
            #     indexes = indexes[:-(abs(count % columns - columns))]
            image_datas = self.parent().parent().imageviewer.model.previews
            self.runIndexes(indexes, image_datas)
            



    def iceOrCarbon(self):
        if not self.currentlyRunning:
            global DISABLE_FUNCTION
            DISABLE_FUNCTION(True)   

              

            image_datas = self.parent().parent().imageviewer.model.previews
            means = []
            medians = []
            for id_ in image_datas:
                if not id_.found_edge:
                    means.append(id_.mean)
                    medians.append(id_.median)
                    # if id_.median > 380:
                    #     id_.original_mask = np.ones_like(id_.original_mask)
            if len(means) > 1:
                self.deciderWindow = carbonIceDecider(self, means)
                self.deciderWindow.show()         
            DISABLE_FUNCTION(False)



            
    
    def runIndexes(self, indexes, image_datas):
        now = datetime.now()
        self.thread = QThread()
        parameters = {}
        for key, (le, t) in self.configLineEdits.items():
            if t is bool:
                parameters[key] = le.isChecked()
            else:
                parameters[key] = t(le.text())
        self.worker = Worker(indexes, image_datas, parameters)
        
        # self.worker.njobs = int(self.njobsLineEdit.text())
        self.number_of_images = len(indexes)
        self.current_number_of_images = 0
        self.parent().parent().setProgress(0)

        # print(f"Creating worker took {(datetime.now() - now).total_seconds()}s")
        now = datetime.now()

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.finishedRunning)
        self.worker.progress.connect(self.testWorker)
        self.thread.start()
        # print(f"Starting worker took {(datetime.now() - now).total_seconds()}s")
        now = datetime.now()

    def testWorker(self, emit):
        global TIMER
        if len(emit) == 1:
            return
        
        idx, mask,thresholdImage, usedImage, hist_data, gridsize,resized_mask, found_edge = emit
        now = datetime.now()
        self.current_number_of_images += 1
        self.parent().parent().setProgress(self.current_number_of_images / self.number_of_images * 100)
        TIMER["progress"] += (datetime.now() - now).total_seconds()
        now = datetime.now()
        view:QTableView = self.parent().parent().imageviewer.view 
        columnCount = self.parent().parent().imageviewer.model.columnCount(None)
        row = idx // columnCount
        col = idx % columnCount
        data:image_data = view.model().index(row,col).data()
        data.mask = q2n.gray2qimage(resized_mask, True) # q2n.array2qimage(mask, True)
        TIMER["q2n"] += (datetime.now() - now).total_seconds()
        now = datetime.now()
        
        data.original_mask = mask
        data.found_edge = found_edge
        TIMER["found_edge"] += (datetime.now() - now).total_seconds()
        now = datetime.now()
        data.hist_data = hist_data
        data.best_gridsize = gridsize
        data.thresholdImage = q2n.gray2qimage(thresholdImage, True)
        data.usedImage = q2n.gray2qimage(usedImage, True)


        index = view.model().createIndex(row, col)
        TIMER["index"] += (datetime.now() - now).total_seconds()
        now = datetime.now()
        view.selectionModel().clearSelection()
        TIMER["clear"] += (datetime.now() - now).total_seconds()
        now = datetime.now()
        view.selectionModel().select(index,QItemSelectionModel.SelectionFlag.Select)
        TIMER["select"] += (datetime.now() - now).total_seconds()
        now = datetime.now()
        view.model().dataChanged.emit(index, index)
        TIMER["emit"] += (datetime.now() - now).total_seconds()
        now = datetime.now()
        # print(row, col)

    def finishedRunning(self):
        global DISABLE_FUNCTION, TIMER
        self.parent().parent().setProgress(100)
        TIMER = defaultdict(lambda:0)
        DISABLE_FUNCTION(False)

    def runSelected(self):
        if not self.currentlyRunning:
            global DISABLE_FUNCTION
            DISABLE_FUNCTION(True)
            idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
            column_count = self.parent().parent().imageviewer.model.columnCount()
            
            new_idxs = [idx.row() * column_count + idx.column() for idx in idxs]

            image_datas = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
            idxs = [(idx.row() * column_count + idx.column()) for idx in idxs]
            self.runIndexes(idxs, image_datas)

    @property
    def currentlyRunning(self):
        global CURRENTLY_RUNNING
        return CURRENTLY_RUNNING
    
    @currentlyRunning.setter
    def currentlyRunning(self, value):
        global CURRENTLY_RUNNING
        CURRENTLY_RUNNING = value

class legendWidget(QWidget):
    def __init__(self, parent):
        global COLORS
        super().__init__(parent)
        self.setLayout(QHBoxLayout())

        self.logoLabel = QLabel(self)
        self_path = Path(__file__).parent.resolve()
        logo_path = self_path / "ced_logo.png"

        logoPixelMap = QPixmap(str(logo_path),).scaled(100,100, Qt.KeepAspectRatio)
        self.logoLabel.setPixmap(logoPixelMap)

        legendLayout = QVBoxLayout()
        self.legends = ((COLORS["not yet"], "Not masked yet"), (COLORS["nothing found"], "No edge was found"), (COLORS["mask found"], "Edge was found"))
        
        self.colorLabels = {}
        # self.descLabels = {}
        for (color, title) in self.legends:
            new_pixmap = QPixmap(10,10)
            new_pixmap.fill(QColor(color))
            newColorLabel = QLabel(self)
            newColorLabel.setPixmap(new_pixmap)

            newDescLabel = QLabel(self, text=title)
            newLayout = QHBoxLayout()
            newPlacerholder = QWidget()
            newLayout.addWidget(newColorLabel)
            newLayout.addWidget(newDescLabel)
            newLayout.addWidget(newPlacerholder,1)
            legendLayout.addLayout(newLayout)
            self.colorLabels[title] = newColorLabel
        self.layout().addLayout(legendLayout)
        self.layout().addWidget(self.logoLabel)

    def loadColors(self):
        global COLORS
        self.legends = ((COLORS["not yet"], "Not masked yet"), (COLORS["nothing found"], "No edge was found"), (COLORS["mask found"], "Edge was found"))
        for (color, title) in self.legends:
            new_pixmap = QPixmap(10,10)
            new_pixmap.fill(QColor(color))
            
            self.colorLabels[title].setPixmap(new_pixmap)


class rightWidget(QWidget):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.legendWidget = legendWidget(self)
        self.histWidget = histWidget(self)
        self.thumbnailWidget = thumbnailWidget(self)
        self.metadataWidget = metadataWidget(self)
        self.runWidget = runWidget(self)
       
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.legendWidget)
        self.layout().addWidget(self.histWidget)
        self.layout().addWidget(self.thumbnailWidget)
        self.layout().addWidget(self.metadataWidget)
        self.layout().addWidget(self.runWidget)
        # self.layout().addWidget(self.logoLabel)
        # self.layout().addWidget(QWidget(),1)
        # self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed))

    def show_data(self, idxs):
        data = self.metadataWidget.show_data(idxs)
        if len(idxs) == 1:
            assert len(data) == 1
            data = data[0]
            self.thumbnailWidget.load_images(data)
            self.histWidget.updatePlot(data, idxs[0])
        else:
            self.thumbnailWidget.clearBoth()
            self.histWidget.clearAll()

    def sizeHint(self):
        # All items the same size.
        return QSize(350, 800)


class QFloatListValidator(QValidator):
    def __init__(self, parent=None, default=None) -> None:
        self.default = default
        super().__init__(parent)
    def validate(self, a0: str, a1: int):
        FloatValidator = QDoubleValidator()
        if len(a0) == 0:
            return (QValidator.State.Intermediate, a0, a1)
        possible_floats = a0.replace(" ", "").split(",")
        correct_floats = []
        for counter, fl in enumerate(possible_floats):
            state, result, _ = FloatValidator.validate(fl, len(fl))
            if state == QValidator.State.Acceptable:
                correct_floats.append(result)
            elif state == QValidator.State.Intermediate and counter == len(possible_floats) - 1:
                correct_floats.append(result)
        if len(correct_floats) == 0:
            return (QValidator.State.Invalid, "", 0)
        result = ", ".join(correct_floats)
        if a0[-1] != " " and result[-1] == " ":
            result = result[:-1]
        return (QValidator.State.Acceptable, result, len(result))

    def fixup(self, a0: str) -> str:
        if len(a0) == 0:
            return str(self.default)
        return super().fixup(a0)





class metadataWidget(QFrame):
    def __init__(self, parent):
        global CURRENT_CONFIG
        super().__init__(parent)
        self.setLayout(QGridLayout())
        self.setFrameShape(QFrame.Panel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        # self.fileNameLabelDesc = QLabel("Filename:")
        # self.fileNameLabel = QLabel("")

        self.dimensionLabelDesc = QLabel("Dimension [px]:")
        # self.dimensionLabel = QLabel("")
        self.dimensionLabel = QLineEdit("")
        self.dimensionLabel.setReadOnly(True)
        

        self.pixelspacingLabelDesc = QLabel("Pixel spacing [Å/px]:")
        self.pixelspacingLabel = QLineEdit("")
        self.pixelspacingLabel.editingFinished.connect(self.setPixelspacing)
        validator = CorrectDoubleValidator(0.001, None, default=1)
        # validator.setBottom(0.001)
        self.pixelspacingLabel.setValidator(validator)
        self.pixelspacingLabel.setToolTip("Pixel spacing in Å per pixel. Usually given by the MRC file header. You can change it if the header is wrong or you are not using mrc files.")

        self.gridsizeLabelDesc = QLabel("Grid hole size [µm]:")
        self.gridsizeLineEdit = QLineEdit("")
        self.gridsizeLineEdit.setToolTip("Size of the grid hole in µm.")

        self.thresholdLabelDesc = QLabel("Threshold")
        self.thresholdLineEdit = QLineEdit("")
        self.thresholdLineEdit.setValidator(CorrectDoubleValidator(None, None, default=CURRENT_CONFIG["parameters"]["threshold"],allow_lists=True))
        self.thresholdLineEdit.editingFinished.connect(self.setThreshold)
        self.thresholdLineEdit.setToolTip("Threshold for finding the edge. Defaul is 0.02. Values to threshold are shown in the histogram after trying to mask the images.")

        self.gridsizeLineEdit.setValidator(QFloatListValidator(default=CURRENT_CONFIG["parameters"]["gridsizes"][0]))
        self.gridsizeLineEdit.setToolTip("Size of the grid hole sizes in µm. If unsure, you can input multiple values seperated by commas and it will try to find the best one.")
        self.gridsizeLineEdit.editingFinished.connect(self.setGridsize)

        self.distanceLabel = QLabel("Distance [Å]")
        self.distanceLineEdit = QLineEdit("")
        self.distanceLineEdit.setValidator(CorrectIntValidator(None, None, default=CURRENT_CONFIG["parameters"]["edge_distance"],allow_lists=True))
        self.distanceLineEdit.setToolTip("Radius inside the hole which should be masked. Put 0 for no masking inside the hole. Using values lower than 0 will inverse the masking. Can be used to mask particles at specific radii in the grid hole.")
        self.distanceLineEdit.editingFinished.connect(self.setDistance)

        self.ringwidthsLabel = QLabel("Return ring width [Å]")
        self.ringwidthsLineEdit = QLineEdit("")
        self.ringwidthsLineEdit.setValidator(CorrectIntValidator(0, None,default=CURRENT_CONFIG["parameters"]["return_ring_width"],allow_lists=True ))
        self.ringwidthsLineEdit.setToolTip("If set to a value greater than 0 only the carbon edge is masked with the given width.")
        self.ringwidthsLineEdit.editingFinished.connect(self.setReturnRingWidth)

        self.outsideLabel = QLabel("Outside of circle coverage [%]")
        self.outsideLineedit = QLineEdit("")
        self.outsideLineedit.setValidator(CorrectDoubleValidator(0,0.999,default=CURRENT_CONFIG["parameters"]["outside_circle_coverage"],allow_lists=True))
        self.outsideLineedit.setToolTip("Minimum percentage of the image which is the grid.")
        self.outsideLineedit.editingFinished.connect(self.setoutsideCoverage)

        self.insideLabel = QLabel("Inside of circle coverage [%]")
        self.insideLineedit = QLineEdit("")
        self.insideLineedit.setValidator(CorrectDoubleValidator(0.001,1,default=CURRENT_CONFIG["parameters"]["inner_circle_coverage"],allow_lists=True))
        self.insideLineedit.setToolTip("Minimum percentage of the image which is the visible hole.")
        self.insideLineedit.editingFinished.connect(self.setinsideCoverage)

        self.highpassLabel = QLabel("High pass filter [Å]")
        self.highpassLineedit = QLineEdit("")
        self.highpassLineedit.setValidator(CorrectDoubleValidator(0,None,default=CURRENT_CONFIG["parameters"]["high_pass_filter"],allow_lists=True))
        self.highpassLineedit.setToolTip("Use a gaussian high pass filter before detecting the grid edge with a size in Å. Can help with large contrast artifacts.")
        self.highpassLineedit.editingFinished.connect(self.sethighpassFilter)


        self.inverseButton = QPushButton("Inverse")
        self.inverseButton.clicked.connect(self.inverse)
        self.inverseButton.setToolTip("Inverse the mask of all selected items.")
        # self.layout().setVerticalSpacing(0)
        # (self.fileNameLabelDesc, self.fileNameLabel)
        for counter, (desc, label) in enumerate([(self.dimensionLabelDesc, self.dimensionLabel),(self.pixelspacingLabelDesc, self.pixelspacingLabel),(self.gridsizeLabelDesc, self.gridsizeLineEdit), (self.thresholdLabelDesc, self.thresholdLineEdit),
                                                 (self.distanceLabel, self.distanceLineEdit), (self.ringwidthsLabel, self.ringwidthsLineEdit), (self.outsideLabel, self.outsideLineedit),
                                                 (self.insideLabel, self.insideLineedit), (self.highpassLabel, self.highpassLineedit)]):

            self.layout().addWidget(desc, counter, 0)
            self.layout().addWidget(label, counter, 1)
            if isinstance(label, QLineEdit) and not label.isReadOnly():
                color = "white"
            else:
                color = "lightgray"
            label.setStyleSheet(f"border: 1px solid black gray;background-color: {color}")
        counter += 1
        self.layout().addWidget(self.inverseButton, counter, 0)
        self.layout().setRowStretch(counter + 1, 1)
        self.layout().setColumnStretch(1,1)
        


    def setoutsideCoverage(self):
        if len(self.outsideLineedit.text().split(" ")) > 1:
            #Warning
            print("Cannot set outside coverage with multiple values.")
            return
        coverage = float(self.outsideLineedit.text())
        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
        for id in ids:
            id.metadata["outside_coverage"] = coverage
        

    def setinsideCoverage(self):
        if len(self.insideLineedit.text().split(" ")) > 1:
            #Warning
            print("Cannot set outside coverage with multiple values.")
            return
        coverage = float(self.insideLineedit.text())
        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
        for id in ids:
            id.metadata["inside_coverage"] = coverage

    def sethighpassFilter(self):

        if len(self.highpassLineedit.text().split(" ")) > 1:
            #Warning
            print("Cannot set outside coverage with multiple values.")
            return
        highpass = float(self.highpassLineedit.text())
        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
        for id in ids:
            id.metadata["high_pass_filter"] = highpass

    def inverse(self):
        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]


        
        rw:runWidget = self.parent().runWidget
        njobs = int(rw.configLineEdits["njobs"][0].text())
        if njobs <= 1 or len(ids) == 1:
            for id in ids:
                if not id.hist_data is None:
                    id:image_data
                    id.metadata["inverse"] = not id.metadata["inverse"]                 
                    hist_data = id.hist_data[id.best_gridsize]
                    # values = hist_data["values"]
                        
                    new_mask = inverseMask(id.original_mask)

                    id.original_mask = new_mask
                    id.found_edge = len(np.unique(new_mask)) > 1
                    id.mask = q2n.gray2qimage(new_mask, True)
        else:
            reevaluateMaskParameters = []
            for id in ids:
                if not id.hist_data is None:
                    id:image_data    
                    id.metadata["inverse"] = not id.metadata["inverse"]             
                    hist_data = id.hist_data[id.best_gridsize]
                    
                    reevaluateMaskParameters.append((id.original_mask, id))

            with mp.get_context("spawn").Pool(njobs) as pool:
                result = [pool.apply_async(inverseMask, [mask], ) for (mask, id) in reevaluateMaskParameters]
                for n, (res, params) in enumerate(zip(result, reevaluateMaskParameters)):
                    id = params[-1]
                    mask = res.get()
                    id.original_mask = mask
                    id.found_edge = len(np.unique(mask)) > 1
                    id.mask = q2n.gray2qimage(mask, True)
                    self.parent().parent().setProgress((n+1)/len(result) * 100,True)
        if len(idxs) == 1:
            self.parent().parent().imageviewer.view.selectionModel().clearSelection()
            self.parent().parent().imageviewer.view.selectionModel().select(idxs[0],QItemSelectionModel.SelectionFlag.Select)
        self.parent().parent().imageviewer.model.layoutChanged.emit()

    def show_data(self, idxs=None):
        # self.fileNameLabel.setText(data.title)
        column_count = self.parent().parent().imageviewer.model.columnCount()
            
        new_idxs = set([idx.row() * column_count + idx.column() for idx in idxs])
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in range(len(self.parent().parent().imageviewer.model.previews)) if idx in new_idxs]
        dimensions = set([str(i.metadata["Dimensions"]) for i in ids])
        pixespacings = set([str(i.metadata["Pixel spacing"]) for i in ids])
        gridsizes = set()
        
        for i in ids:
            gridsizes.update([str(gs) for gs in i.metadata["Gridsize"]])
        # gridsizes = set([str(i.metadata["Gridsize"]) for i in ids])
        thresholds = set([str(i.metadata["Threshold"]) for i in ids])
        distances = set([str(i.metadata["distance"]) for i in ids])
        ring_widths = set([str(i.metadata["return_ring_width"]) for i in ids])
        outside_coverages = set([str(i.metadata["outside_coverage"]) for i in ids])
        inside_coverages = set([str(i.metadata["inside_coverage"]) for i in ids])
        high_pass_filters = set([str(i.metadata["high_pass_filter"]) for i in ids])
        self.dimensionLabel.setText(", ".join(dimensions))
        self.pixelspacingLabel.setText(", ".join(pixespacings))
        self.gridsizeLineEdit.setText(", ".join(gridsizes))
        self.thresholdLineEdit.setText(", ".join(thresholds))
        self.distanceLineEdit.setText(", ".join(distances))
        self.ringwidthsLineEdit.setText(", ".join(ring_widths))
        self.outsideLineedit.setText(", ".join(outside_coverages))
        self.insideLineedit.setText(", ".join(inside_coverages))
        self.highpassLineedit.setText(", ".join(high_pass_filters))
        return ids
        # else:
        #     self.dimensionLabel.setText(str(data.metadata["Dimensions"]))
        #     self.pixelspacingLabel.setText(str(data.metadata["Pixel spacing"]))
        #     self.gridsizeLineEdit.setText(", ".join([str(i) for i in data.metadata["Gridsize"]]))

    def setReturnRingWidth(self, event=None):
        if len(self.ringwidthsLineEdit.text().split(" ")) > 1:
            #Warning
            print("Cannot set return ring width with multiple values.")
            return
        returnRingWidth = int(self.ringwidthsLineEdit.text())
        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
        for id in ids:
            id.metadata["return_ring_width"] = returnRingWidth
        self.setThreshold(None)

    def setDistance(self, event=None):
        if len(self.distanceLineEdit.text().split(" ")) > 1:
            #Warning
            print("Cannot set distance with multiple values.")
            return
        distance = int(self.distanceLineEdit.text())
        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
        for id in ids:
            id.metadata["distance"] = distance
        self.setThreshold(None)

    def setGridsize(self, event=None):
        
        gridsizes = self.gridsizeLineEdit.text().replace(" ", "").split(",")
        new_gridsizes = [float(gs) for gs in gridsizes if len(gs) > 0]

        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
        for id in ids:
            id.metadata["Gridsize"] = new_gridsizes
        

    def setPixelspacing(self, event=None):
        if len(self.pixelspacingLabel.text().split(" ")) > 1:
            #Warning
            print("Cannot set pixel spacing with multiple values.")
            return
        pixel_spacing = float(self.pixelspacingLabel.text())
        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
        for id in ids:
            id.metadata["Pixel spacing"] = pixel_spacing

    def setThreshold(self, event=None):
        if len(self.thresholdLineEdit.text().split(" ")) > 1:
            #Warning
            print("Cannot set threshold with multiple values.")
            return
        threshold = float(self.thresholdLineEdit.text())
        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]


        
        rw:runWidget = self.parent().runWidget
        njobs = int(rw.configLineEdits["njobs"][0].text())
        if njobs <= 1 or len(ids) == 1:
            for id in ids:
                id.metadata["Threshold"] = threshold
                if not id.hist_data is None:
                    id:image_data                 
                    hist_data = id.hist_data[id.best_gridsize]
                    edges = hist_data["edges"]
                    # values = hist_data["values"]
                    if np.max(edges) >= threshold:
                        
                        new_mask = reevaluateMask(hist_data["center"], id.best_gridsize, id.metadata["Pixel spacing"],id.metadata["Dimensions"], id.metadata["distance"], id.metadata["return_ring_width"], id.metadata["inverse"])
                    else:
                        new_mask = np.ones(id.metadata["Dimensions"], np.uint8)
                    id.original_mask = new_mask
                    id.found_edge = len(np.unique(new_mask)) > 1
                    id.mask = q2n.gray2qimage(new_mask, True)
        else:
            reevaluateMaskParameters = []
            for id in ids:
                id.metadata["Threshold"] = threshold
                if not id.hist_data is None:
                    id:image_data                 
                    hist_data = id.hist_data[id.best_gridsize]
                    edges = hist_data["edges"]
                    reevaluateMaskParameters.append((hist_data["center"], id.best_gridsize, id.metadata["Pixel spacing"],id.metadata["Dimensions"], threshold, np.max(edges), id, id.metadata["distance"], id.metadata["return_ring_width"], id.metadata["inverse"]))

            with mp.get_context("spawn").Pool(njobs) as pool:
                result = [pool.apply_async(reevaluateMaskMP, [center, gridsize, ps, shape, threshold, max_, distance, ring_width, inverse], ) for (center, gridsize, ps, shape, threshold, max_, id, distance, ring_width, inverse) in reevaluateMaskParameters]
                for n, (res, params) in enumerate(zip(result, reevaluateMaskParameters)):
                    id = params[-4]
                    mask, found = res.get()
                    id.original_mask = mask
                    id.found_edge = found
                    id.mask = q2n.gray2qimage(mask, True)
                    self.parent().parent().setProgress((n+1)/len(result) * 100,True)
        if len(idxs) == 1:
            self.parent().parent().imageviewer.view.selectionModel().clearSelection()
            self.parent().parent().imageviewer.view.selectionModel().select(idxs[0],QItemSelectionModel.SelectionFlag.Select)
        self.parent().parent().imageviewer.model.layoutChanged.emit()



class mainWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.imageviewer = ImageViewer(self)
        self.metadataWidget = rightWidget(self)
        self.progressBar = QProgressBar(self)
        self.progressBar.setMaximum(100)
        self.progressBar.setMinimum(0)
        self.setLayout(QVBoxLayout())
        self.lowerLayout = QHBoxLayout()
        
        # self.test_button = QPushButton(text="test")
        # self.test_button.clicked.connect(self.pushed_button)
        # self.layout().addWidget(self.test_button)

        self.layout().addWidget(self.progressBar)
        self.layout().addLayout(self.lowerLayout)
        self.lowerLayout.addWidget(self.imageviewer)
        self.lowerLayout.addWidget(self.metadataWidget)
        self.setProgress(0)

    def pushed_button(self):
        self.imageviewer.load_files()

    def setProgress(self, progress, update=False):
        progress = round(progress)
        self.progressBar.setValue(progress)
        if update:
            QApplication.processEvents()

    def sizeHint(self):
        # All items the same size.
        return QSize(1600, 1000)

class MainWindow(QMainWindow):
    def __init__(self, custom_parent=None):
        super().__init__()

        global CURRENT_CONFIG, DISABLE_FUNCTION
        global COLORS_ALTERNATIVE, COLORS, COLORS_DEFAULT
        CURRENT_CONFIG = load_config_file()
        if CURRENT_CONFIG["misc"]["colorblind_mode"]:
            COLORS = COLORS_ALTERNATIVE
        else:
            COLORS = COLORS_DEFAULT

        ray.init()



        self.customParent=custom_parent
        self.mainwidget = mainWidget(self)
        self.createMenu()
        self.setCentralWidget(self.mainwidget)
        self.setWindowTitle("Grid edge detector")

        DISABLE_FUNCTION = getDisableEverythingFunction(self)
    
    def createMenu(self):
        global CURRENT_CONFIG
        self.menuBar().clear()
        menuBar = self.menuBar()
        self.filesMenu = QMenu("Files", menuBar)
        self.loadFilesAction = self.filesMenu.addAction("Load files")
        self.importMenuBar = QMenu("Import", self.filesMenu)
        self.filesMenu.addMenu(self.importMenuBar)
        self.exportMenuBar = QMenu("Export", self.filesMenu)
        self.filesMenu.addMenu(self.exportMenuBar)
        try:
            from cryovia.gui.dataset import Dataset, get_all_dataset_names

            self.importDatasetMenu = self.importMenuBar.addMenu("CryoVIA dataset")
            self.exportDatasetMenu = self.exportMenuBar.addMenu("CryoVIA dataset")
            self.exportDatasetMenu.aboutToShow.connect(self.openedExportDataset)
            self.importDatasetMenu.aboutToShow.connect(self.openedImportDataset)
            # self.exportDatasetMenu.hovered.connect(self.openedExportDataset)
            

        except Exception as e:
            self.importDataset = self.importMenuBar.addMenu("CryoVia dataset")
            self.exportDatasetMenu = self.exportMenuBar.addMenu("CryoVia dataset")
            self.importDataset.setEnabled(False)
            self.exportDatasetMenu.setEnabled(False)
        
        # CryoSparc Project Import

        if len(CURRENT_CONFIG["files"]["cryosparc_projects_dir"]) > 0 and Path(CURRENT_CONFIG["files"]["cryosparc_projects_dir"]).exists():
            self.importCryoSparcProject = self.importMenuBar.addMenu("CryoSparc project")
            self.importCryoSparcProject.aboutToShow.connect(self.setCryoSparcMenu)
        else:
            act = self.importMenuBar.addAction("Set CryoSparc project folder")
            act.triggered.connect(self.setCryoSparcProjectFolder)
        self.importRlnPicking = self.importMenuBar.addAction("Import Relion picking job")
        self.importRlnPicking.triggered.connect(self.loadRlnPicking)
        
        self.exportRlnPicking = self.exportMenuBar.addMenu("Relion picking jobs")
        self.exportRlnPicking.aboutToShow.connect(self.openedExportRlnPickingJobs)
        
        
        self.exportCryoSparcProjectMenu = self.exportMenuBar.addMenu("CryoSparc project")
        self.exportCryoSparcProjectMenu.aboutToShow.connect(self.openedExportCryoSparcProject)



        self.saveFilesAction = self.filesMenu.addAction("Save all masks")
        self.saveSelectedFilesAction = self.filesMenu.addAction("Save selected masks")
        self.saveMaskedImagesMenu = self.filesMenu.addMenu("Save all masked images")
        self.saveSelectedMaskedImagesMenu = self.filesMenu.addMenu("Save selected masked images")
        menuBar.addMenu(self.filesMenu)
        
        self.loadFilesAction.triggered.connect(self.loadFiles)
        
        self.saveFilesAction.triggered.connect(self.saveAllMasks)
        self.saveSelectedFilesAction.triggered.connect(self.saveSelectedMasks)
        # self.saveMaskedImagesAction.triggered.connect(self.saveAllMaskedImages)

        for (name, method) in [("Minimum", "min"), ("Maximum","max"),("Mean","mean"),("Random noise","random noise")]:
            act = self.saveSelectedMaskedImagesMenu.addAction(name)
            act.triggered.connect(lambda state, x=method: self.saveSelectedMaskedImages(x))

            act = self.saveMaskedImagesMenu.addAction(name)
            act.triggered.connect(lambda state, x=method: self.saveAllMaskedImages(x))
        self.saveSelectedCsvAction = self.filesMenu.addAction("Save CSV-file for selected images")
        self.saveAllCsvAction = self.filesMenu.addAction("Save CSV-file for all images")

        self.saveSelectedCsvAction.triggered.connect(self.saveSelectedCsv)
        self.saveAllCsvAction.triggered.connect(self.saveAllCsv)

        # self.saveSelectedMaskedImagesAction.triggered.connect(self.saveSelectedMaskedImages)
        
        self.configMenu = QMenu("Config", menuBar)
        self.openConfigFileAction = self.configMenu.addAction("Open config file")
        # self.createDefaultConfigFileAction = self.configMenu.addAction("Create default config file")
        self.toggleColorblindModeAction = self.configMenu.addAction("Toggle colorblind mode")
        self.openConfigFileAction.triggered.connect(self.openConfigFile)
        # self.createDefaultConfigFileAction.triggered.connect(self.createDefaultConfig)
        self.toggleColorblindModeAction.triggered.connect(self.toggleColorblindMode)
        menuBar.addMenu(self.configMenu)



    def setCryoSparcProjectFolder(self):
        global CURRENT_CONFIG, CONFIG_FILE
        dlg = QFileDialog()

        dlg.setFileMode(QFileDialog.ExistingFiles)
        folder = dlg.getExistingDirectory(self, "Choose dataset directory", ".")
        if folder is None or len(folder) == 0:
            return
        CURRENT_CONFIG["files"]["cryosparc_projects_dir"] = folder
        if check_config(CURRENT_CONFIG):
            with open(CONFIG_FILE, "w") as f:
                toml.dump(CURRENT_CONFIG,f)
        self.createMenu()

    def setCryoSparcMenu(self):
        global CURRENT_CONFIG, TIMER
        self.importCryoSparcProject.clear()
        project_dir = Path(CURRENT_CONFIG["files"]["cryosparc_projects_dir"])
        for project in os.listdir(project_dir):
            current_dir = project_dir / project
            if checkCrypSparcProject(current_dir):
                now = datetime.now()
                menu = self.importCryoSparcProject.addMenu(project)
                TIMER["createMenus"] += (datetime.now() - now).total_seconds()
                now = datetime.now()
                jobs = findPickingJobs(current_dir)
                TIMER["findPickingJobsTotal"] += (datetime.now() - now).total_seconds()
                now = datetime.now()
                for job in jobs:
                    
                    act = menu.addAction(job)
                    act.triggered.connect(lambda state, x=current_dir / job: self.loadCryoSparcJob(x) )
                TIMER["createActions"] += (datetime.now() - now).total_seconds()
                now = datetime.now()


    def loadCryoSparcJob(self, jobPath:Path):
        project_folder = jobPath.parent
        ds = Dataset.load(jobPath / "picked_particles.cs")
        paths = sorted(list(set(ds["location/micrograph_path"])))
        particles = {}
        path_translator = {}
        for key in paths:
            current_file = project_folder / key
            if current_file.is_symlink():
                current_file = current_file.readlink()
            particles[current_file] = {"x":[], "y":[], "inner_path":key, "project_path":project_folder, "job_path":jobPath}
            path_translator[key] = current_file


        for x,y, m in zip(ds["location/center_x_frac"], ds["location/center_y_frac"], ds["location/micrograph_path"]):
            particles[path_translator[m]]["x"].append(x)
            particles[path_translator[m]]["y"].append(y)



        ds = Dataset.load(jobPath / "picked_micrographs.cs")
        for ps, m, shape in zip(ds["micrograph_blob/psize_A"], ds["micrograph_blob/path"], ds["micrograph_blob/shape"]):
            
            particles[path_translator[m]]["ps"] = ps
            particles[path_translator[m]]["shape"] = shape
        
        files = [key for key in sorted(particles.keys())]
        pixelSizes = {key:value["ps"] for key, value in particles.items()}
        self.mainwidget.imageviewer.load_files(files, pixelSizes=pixelSizes, cryoSparcInfo=particles)

    def loadRlnPicking(self):
        filename, filt = QFileDialog.getOpenFileName(self, "Select a picking star file", CURRENT_CONFIG["files"]["filedir"], "Star file (autopick.star manualpick.star)")
        if filename:
            self.loadRelionJob(filename)



    def loadRelionJob(self, pickPath):
        def extract_coordinates(sf):
            coordinates_df = starfile.read(sf)
            x = coordinates_df["rlnCoordinateX"].astype(np.int32)
            y = coordinates_df["rlnCoordinateY"].astype(np.int32)
            return y,x

        pick_file = Path(pickPath)
        project_file = pick_file.parent.parent.parent
        autopick_df = starfile.read(pick_file)

        rlnInfo = {}
        files = []


        for i in range(len(autopick_df)):

            absolut_path = project_file/ autopick_df.loc[i]["rlnMicrographName"]
            rlnInfo[absolut_path] = {"coordinate_file": autopick_df.loc[i]["rlnMicrographCoordinates"]}
            rlnInfo[absolut_path]["absolute_coordinate_file"] = project_file/ autopick_df.loc[i]["rlnMicrographCoordinates"]
            rlnInfo[absolut_path]["mrc_file"] = autopick_df.loc[i]["rlnMicrographName"]

            y,x = extract_coordinates(rlnInfo[absolut_path]["absolute_coordinate_file"])

            rlnInfo[absolut_path]["y"] = y
            rlnInfo[absolut_path]["x"] = x
            rlnInfo[absolut_path]["pickPath"] = pick_file
            files.append(absolut_path)
        self.mainwidget.imageviewer.load_files(files, pixelSizes=None, rlnInfo=rlnInfo)


    def openedExportRlnPickingJobs(self):
        jobs = set()
        for img_data in self.mainwidget.imageviewer.model.previews:
            img_data:image_data
            if img_data.rlnInfo is not None:
                # project = img_data.cryosparc_info["project_path"]
                pick_path = str(img_data.rlnInfo["pickPath"])
                jobs.add(pick_path)
        self.exportRlnPicking.clear()

        for job in jobs:
            act = self.exportRlnPicking.addAction(job)
            act.triggered.connect(lambda state, x=job:self.exportRlnJob(x))
           

    def exportRlnJob(self, job):
         
        img_data_to_use = []
        coordinate_dir = Path(job).parent

        for img_data in self.mainwidget.imageviewer.model.previews:
            img_data:image_data
            if img_data.rlnInfo is not None:
                if str(img_data.rlnInfo["pickPath"]) == str(job):
                    img_data_to_use.append(img_data)

        

        
        for img_data in img_data_to_use:
            idxs = []
            img_data:image_data
            if img_data.found_edge:

                # masks[img_data.rlnInfo["rlnMicrographCoordinates"]] = img_data.original_mask
                for idx, (y,x) in enumerate(zip(img_data.rlnInfo["y"],img_data.rlnInfo["x"])):
                    if img_data.original_mask[y,x] == 1:
                        idxs.append(idx)
                coordinate_file = img_data.rlnInfo["absolute_coordinate_file"]
                coordinate_df = starfile.read(coordinate_file)
                coordinate_df = coordinate_df.iloc[idxs]
            starfile.write(coordinate_df, img_data.rlnInfo["absolute_coordinate_file"],)
                

    def openedImportDataset(self):
        from cryovia.gui.dataset import Dataset, get_all_dataset_names
        self.importDatasetMenu.clear()
        names = sorted(list(get_all_dataset_names()))
        for name in names:
            act = self.importDatasetMenu.addAction(name)
            act.triggered.connect(lambda state, x=name: self.loadDataset(x) )


    def openedExportCryoSparcProject(self):
        projects = {}
        for img_data in self.mainwidget.imageviewer.model.previews:
            img_data:image_data
            if img_data.cryosparc_info is not None:
                project = img_data.cryosparc_info["project_path"]
                job = img_data.cryosparc_info["job_path"]
                if project not in projects:
                    projects[project] = set()
                projects[project].add(job)
        self.exportCryoSparcProjectMenu.clear()

        for project, jobs in projects.items():
            projectmenu:QMenu = self.exportCryoSparcProjectMenu.addMenu(project.name)
            for job in jobs:
                act = projectmenu.addAction(job.name)
                act.triggered.connect(lambda state, x=project, y=job:self.exportCryoSparcProject(x, y))


    def exportCryoSparcProject(self, project, job):
        img_data_to_use = []
        for img_data in self.mainwidget.imageviewer.model.previews:
            img_data:image_data
            if img_data.cryosparc_info is not None:
                if img_data.cryosparc_info["job_path"] == job:
                    img_data_to_use.append(img_data)

        ds = Dataset.load(job / "picked_particles.cs")

        masks = {}
        shapes = {}
        for img_data in img_data_to_use:
            img_data:image_data
            if img_data.found_edge:

                masks[img_data.cryosparc_info["inner_path"]] = img_data.original_mask
                shapes[img_data.cryosparc_info["inner_path"]] = img_data.cryosparc_info["shape"]
            else:
                masks[img_data.cryosparc_info["inner_path"]] = None
                shapes[img_data.cryosparc_info["inner_path"]] = None
        idxs = []
        for counter, (x,y, m) in enumerate(zip(ds["location/center_x_frac"], ds["location/center_y_frac"], ds["location/micrograph_path"])):
            if m in masks and masks[m] is not None:
                shape = shapes[m]
                
                x = int(x*shape[1])
                y = int(y*shape[0])
                if masks[m][y,x] == 1:
                    idxs.append(True)
                else:
                    idxs.append(False)
            else:
                idxs.append(True)
        ds = ds.mask(idxs)
        shutil.move(job / "picked_particles.cs", job / "picked_particles_old.cs")
        ds.save(job / "picked_particles.cs")
        
    def openedExportDataset(self):

        datasets = set()
        for img_data in self.mainwidget.imageviewer.model.previews:
            if img_data.dataset is not None:
                datasets.add(img_data.dataset) 
        
        
        self.exportDatasetMenu.clear()
        
        for name in datasets:
            act = self.exportDatasetMenu.addAction(name)
            act.triggered.connect(lambda state, x=name:self.exportDataset(name))

    def exportDataset(self, dataset):
        global CURRENTCOUNTER
        
        def getCallback(maxNumber):
            def callback(result):
                global CURRENTCOUNTER
                CURRENTCOUNTER += 1
                self.mainwidget.setProgress(CURRENTCOUNTER/maxNumber * 100, True)
            return callback
        from cryovia.gui.dataset import Dataset
        from cryovia.cryovia_analysis.analyser import Analyser, AnalyserWrapper
        global MAX_CORES
        dataset:Dataset = Dataset.load(dataset)
        number_of_files = 0
        for img_data in self.mainwidget.imageviewer.model.previews:
            if img_data.dataset is not None and img_data.dataset == dataset.name:
                img_data:image_data
                if img_data.original_mask is not None:
                    if Path(img_data.fn) in dataset.analysers:
                       number_of_files += 1
                    elif str(img_data.fn) in dataset.analysers:
                        number_of_files += 1


        mask_path = dataset.mask_path
        for img_data in self.mainwidget.imageviewer.model.previews:
            if img_data.dataset is not None and img_data.dataset == dataset.name:
                img_data:image_data
                if img_data.original_mask is not None :
                    current_path_name = mask_path / (Path(img_data.fn).stem + "_mask.pickle")
                    hist_data = img_data.hist_data[img_data.best_gridsize]
                   
                        
                    mask_data = mask_file(img_data.metadata["Dimensions"], hist_data["center"], img_data.best_gridsize,img_data.metadata["distance"],img_data.metadata["return_ring_width"], 
                                          img_data.metadata["inverse"], img_data.fn, img_data.metadata["Pixel spacing"], img_data.found_edge, np.max(img_data.original_mask)==1  )
                    
                    with open(current_path_name, "wb") as f:
                        pickle.dump(mask_data, f)



    def loadDataset(self, name):
        global DISABLE_FUNCTION
        from cryovia.gui.dataset import Dataset
        dataset:Dataset = Dataset.load(name)
        files = dataset.micrograph_paths
        if len(files) > 0:
            try:
                DISABLE_FUNCTION(True)
                self.mainwidget.imageviewer.load_files(files, dataset)
                DISABLE_FUNCTION(False)
            except Exception as e:
                e = traceback.format_exc()
                print(e)                
                DISABLE_FUNCTION(False)



    def openConfigFile(self):
        global CONFIG_FILE, CURRENT_CONFIG
        # if platform.system() == 'Darwin':       # macOS
        #     subprocess.call(('open', CONFIG_FILE))
        # elif platform.system() == 'Windows':    # Windows
        #     os.startfile(CONFIG_FILE)
        # else:                                   # linux variants
        #     subprocess.call(('xdg-open', CONFIG_FILE))
        self.ConfigWindow = ConfigWindow(self)
        
        self.ConfigWindow.show()
    
    def loadConfig(self):
        global CURRENT_CONFIG
        CURRENT_CONFIG = load_config_file()
        previews = self.mainwidget.imageviewer.model.previews
        for i in previews:
            i:image_data 
            i.metadata["Gridsize"] = CURRENT_CONFIG["parameters"]["gridsizes"]
            i.metadata["Threshold"] = CURRENT_CONFIG["parameters"]["threshold"]
            i.metadata["distance"] = CURRENT_CONFIG["parameters"]["edge_distance"]
            i.metadata["return_ring_width"] = CURRENT_CONFIG["parameters"]["return_ring_width"]
            i.metadata["outside_coverage"] = CURRENT_CONFIG["parameters"]["outside_circle_coverage"]
            i.metadata["inside_coverage"] = CURRENT_CONFIG["parameters"]["inner_circle_coverage"]
            i.metadata["high_pass_filter"] = CURRENT_CONFIG["parameters"]["high_pass_filter"]

        self.mainwidget.metadataWidget.runWidget:runWidget
        for key, value in CURRENT_CONFIG["parameters"].items():
            if key in self.mainwidget.metadataWidget.runWidget.configLineEdits: 
                self.mainwidget.metadataWidget.runWidget.configLineEdits[key][0].setText(str(value))
            else:
                pass
        # self.mainwidget.metadataWidget.runWidget.njobsLineEdit.setText(str(CURRENT_CONFIG["parameters"]["njobs"]))
        # self.mainwidget.metadataWidget.runWidget.resizeLineEdit.setText(str(CURRENT_CONFIG["parameters"]["resize_value"]))
        # self.mainwidget.metadataWidget.runWidget.toggleResizeCheckbox.setChecked(CURRENT_CONFIG["parameters"]["resize_bool"])
        self.loadColorBlindMode()
        # self.mainwidget.metadataWidget.runWidget.thresholdLineEdit.setText(str(CURRENT_CONFIG["parameters"]["threshold"]))

    # def createDefaultConfig(self):
    #     global CURRENT_CONFIG
    #     create_default_config_file(True)
    #     CURRENT_CONFIG = load_config_file()
    #     self.loadConfig()


    def setDisabledActions(self, disable=True):
        global DISABLE_FUNCTION
        DISABLE_FUNCTION(disable)
        # self.loadFilesAction.setDisabled(disable)
        # self.saveFilesAction.setDisabled(disable)
        # self.saveSelectedFilesAction.setDisabled(disable)
        # self.mainwidget.metadataWidget.runWidget.runAllButton.setDisabled(disable)
        # self.mainwidget.metadataWidget.runWidget.runSelectedButton.setDisabled(disable)

    def loadFiles(self):
        global DISABLE_FUNCTION
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        file_suffixes = sorted([f"*{i}" for i in ALLOWED_SUFFIXES])
        file_suffixes = " *".join(file_suffixes)
        file_suffixes = f"mrc files (*{file_suffixes})"
        names, filt = dialog.getOpenFileNames(self, "Open images", CURRENT_CONFIG["files"]["filedir"],file_suffixes)
        # dialog.getOpenFileNames(self, "Open images", "/home", )
        # filedialog = QFileDialog.getOpenFileName(self, "Open Image", "/home", "")
        names = [Path(name) for name in names]
        DISABLE_FUNCTION(True)
        self.mainwidget.imageviewer.load_files(names)
        DISABLE_FUNCTION(False)
        

    def saveAllMasks(self):
        image_datas = self.mainwidget.imageviewer.model.previews
        self.saveMasks(image_datas)
    

    def saveAllCsv(self):
        image_datas = self.mainwidget.imageviewer.model.previews
        self.saveCsvFile(image_datas)
    


    def saveSelectedCsv(self):
        idxs = self.mainwidget.imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.mainwidget.imageviewer.model.columnCount() + idx.column() for idx in idxs]
        image_datas = [self.mainwidget.imageviewer.model.previews[idx] for idx in new_idxs]
        self.saveCsvFile(image_datas)


    def saveSelectedMasks(self):
        
        idxs = self.mainwidget.imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.mainwidget.imageviewer.model.columnCount() + idx.column() for idx in idxs]
        image_datas = [self.mainwidget.imageviewer.model.previews[idx] for idx in new_idxs]
        self.saveMasks(image_datas)

    def saveSelectedMaskedImages(self, method="min"):
        idxs = self.mainwidget.imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.mainwidget.imageviewer.model.columnCount() + idx.column() for idx in idxs]
        image_datas = [self.mainwidget.imageviewer.model.previews[idx] for idx in new_idxs]
        self.saveMaskedImages(image_datas,method=method)

    def saveAllMaskedImages(self, method="min"):
        image_datas = self.mainwidget.imageviewer.model.previews
        self.saveMaskedImages(image_datas, method=method)


    def saveCsvFile(self, image_datas):
        global CURRENT_CONFIG  
        dialog = QFileDialog()
        
        files = []
        found_carbon = []
        percentages = []
        save_dir = dialog.getExistingDirectory(self, "Save directory", CURRENT_CONFIG["files"]["filedir"])
        if save_dir is not None and len(save_dir) > 0:
            with open(Path(save_dir) / "test.csv", "w") as f:
                f.write("file\tfound_edge\tpercentage_of_image\n")
                for id in image_datas:
                    files.append(id.fn)
                    if id.found_edge is None:
                        found_carbon.append(np.nan)
                    else:
                        found_carbon.append(id.found_edge)
                    percentages.append(id.percentage)


                sorted_index = np.argsort(percentages)
                for idx in sorted_index:
                    f.write(f"{files[idx]}\t{found_carbon[idx]}\t{percentages[idx]}\n")
        


    def saveMaskedImages(self, image_datas, method="min"):
        global CURRENT_CONFIG  
        dialog = QFileDialog()
        # dialog.setFileMode(QFileDialog.Ex)
        save_dir = dialog.getExistingDirectory(self, "Save directory", CURRENT_CONFIG["files"]["filedir"])
        if save_dir is not None and len(save_dir) > 0:
            save_dir = Path(save_dir).absolute()
            for id in image_datas:
                if id.original_mask is not None:
                    current_path = save_dir / (id.fn.stem + CURRENT_CONFIG["files"]["masked_image_file_suffix"] + id.fn.suffix)
                    data = id.original_image
                    if method == "min":
                        data[id.original_mask == 0] = np.min(data[id.original_mask != 0])
                    elif method == "max":
                        data[id.original_mask == 0] = np.max(data[id.original_mask != 0])
                    elif method == "mean":
                        data[id.original_mask == 0] = np.mean(data[id.original_mask != 0])
                    elif method == "random noise":
                        mean = np.mean(data[id.original_mask != 0])
                        std = np.std(data[id.original_mask != 0])
                        noise = np.random.normal(mean, std, size = (data[id.original_mask == 0]).size)
                        data[id.original_mask == 0] = noise
                    if current_path.suffix in [".mrc", ".MRC", ".rec", ".REC"]:
                        with mrcfile.new(current_path, data=data, overwrite=True) as f:
                            f.voxel_size = id.metadata["Pixel spacing"]
                    else:
                        plt.imsave(data, id.original_mask, cmap="gray") 



    def saveMasks(self, image_datas):
        global CURRENT_CONFIG
        
        dialog = QFileDialog()
        # dialog.setFileMode(QFileDialog.Ex)
        save_dir = dialog.getExistingDirectory(self, "Save directory", CURRENT_CONFIG["files"]["filedir"])
        if save_dir is not None and len(save_dir) > 0:
            save_dir = Path(save_dir).absolute()
            for id in image_datas:
                if id.original_mask is not None:
                    current_path = save_dir / (id.fn.stem + CURRENT_CONFIG["files"]["mask_file_suffix"] + id.fn.suffix)
                    if current_path.suffix in [".mrc", ".MRC", ".rec", ".REC"]:
                        with mrcfile.new(current_path, data=id.original_mask, overwrite=True) as f:
                            f.voxel_size = id.metadata["Pixel spacing"]
                    else:
                        plt.imsave(current_path, id.original_mask, cmap="gray")    
        
    def toggleColorblindMode(self):
        global CURRENT_CONFIG, COLORS, COLORS_ALTERNATIVE, COLORS_DEFAULT
        CURRENT_CONFIG["misc"]["colorblind_mode"] = not CURRENT_CONFIG["misc"]["colorblind_mode"]
        self.loadColorBlindMode()

    def loadColorBlindMode(self):
        global CURRENT_CONFIG, COLORS, COLORS_ALTERNATIVE, COLORS_DEFAULT
        if CURRENT_CONFIG["misc"]["colorblind_mode"]:
            COLORS = COLORS_ALTERNATIVE
        else:
            COLORS = COLORS_DEFAULT
        self.mainwidget.metadataWidget.legendWidget.loadColors() 
        self.mainwidget.imageviewer.model.layoutChanged.emit()
    

    def resizeEvent(self, a0) -> None:

        # self.model.modelReset.emit()
        self.mainwidget.imageviewer.model.layoutChanged.emit()
        self.mainwidget.imageviewer.view.resizeColumnsToContents()
        self.mainwidget.imageviewer.view.resizeRowsToContents()
        return super().resizeEvent(a0)

    def closeEvent(self, a0) -> None:
        if self.customParent is not None:
            ray.shutdown()
            self.customParent.child_closed()
        return super().closeEvent(a0)


class ConfigWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parentWindow = parent
        self.file_path = None

        self.save_current_file_shortcut = QShortcut(QKeySequence('Ctrl+S'), self)
        self.save_current_file_shortcut.activated.connect(self.save_current_file)

        vbox = QVBoxLayout()
        
        self.title = QLabel("")
        self.title.setWordWrap(True)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.loadDefaultButton = QPushButton("Load default values")
        self.loadConfigFileButton = QPushButton("Load config file")
        self.saveFileButton = QPushButton("Save")
        self.applyButton = QPushButton("Save and apply")

        self.loadDefaultButton.clicked.connect(self.loadDefaultConfig)
        self.loadConfigFileButton.clicked.connect(self.open_new_file)
        self.saveFileButton.clicked.connect(self.save_current_file)
        self.applyButton.clicked.connect(self.apply)
       
        self.titleLayout = QHBoxLayout()
        self.titleLayout.addWidget(self.loadDefaultButton)
        self.titleLayout.addWidget(self.loadConfigFileButton)
        self.titleLayout.addWidget(self.saveFileButton)
        self.titleLayout.addWidget(self.applyButton)
        
        
        vbox.addWidget(self.title)
        self.setLayout(vbox)
        self.layout().addLayout(self.titleLayout)

        self.scrollable_text_area = QTextEdit()
        vbox.addWidget(self.scrollable_text_area)

        self.changed_ = False

        self.scrollable_text_area.textChanged.connect(self.somethingChanged)

        self.open_new_file()


    @property
    def changed(self):
        return self.changed_

    @changed.setter
    def changed(self, value):
        if self.changed and value:
            return
        if self.changed and not value:
            self.changed_ = value
            self.title.setText(self.title.text()[:-1])
            return
        if not self.changed and value:
            self.changed_ = value
            self.title.setText(self.title.text() + "*")
        if not self.changed and not value:
            return

    def somethingChanged(self, event=None):
        self.changed = True        

    def loadDefaultConfig(self):
        global DEFAULT_CONFIG
        config = toml.dumps(DEFAULT_CONFIG)
        self.scrollable_text_area.setText(config)

    def open_new_file(self):
        global CONFIG_FILE 
        
        if CONFIG_FILE:
            try:
                config = toml.load(CONFIG_FILE)
                config_str = toml.dumps(config)
            
                self.title.setText(str(CONFIG_FILE))
                self.scrollable_text_area.setText(config_str)
            except:
                create_default_config_file(True)
                config = toml.load(CONFIG_FILE)
                config_str = toml.dumps(config)
            
                self.title.setText(str(CONFIG_FILE))
                self.scrollable_text_area.setText(config_str)
            self.changed = False
        else:
            self.invalid_path_alert_message()

    def save_current_file(self):
        global CONFIG_FILE
        if not CONFIG_FILE:
            
            self.invalid_path_alert_message()
            return False
        file_contents = self.scrollable_text_area.toPlainText()
        try:
            config = toml.loads(file_contents)
            if check_config(config):
                with open(CONFIG_FILE, "w") as f:
                    toml.dump(config,f)
                self.changed = False
                
            else:
                self.invalid_toml_file_message()
                return False
               
        except:
            self.invalid_toml_file_message()
            return False
        return True
            
    
    def closeEvent(self, event):
        if self.changed:
            messageBox = QMessageBox()
            title = "Quit Application?"
            message = "WARNING !!\n\nIf you quit without saving, any changes made to the file will be lost.\n\nSave file before quitting?"
        
            reply = messageBox.question(self, title, message, messageBox.Yes | messageBox.No |
                    messageBox.Cancel, messageBox.Cancel)
            if reply == messageBox.Yes:
                return_value = self.save_current_file()
                if return_value == False:
                    event.ignore()
            elif reply == messageBox.No:
                
                event.accept()
            else:
                event.ignore()
        else:
            
            event.accept()
    
    def apply(self):
        if self.save_current_file():
        
            self.parentWindow.loadConfig()

    def invalid_path_alert_message(self):
        messageBox = QMessageBox()
        messageBox.setWindowTitle("Invalid file")
        messageBox.setText("Selected filename or path is not valid. Please select a valid file.")
        messageBox.exec()


    def invalid_toml_file_message(self):
        messageBox = QMessageBox()
        messageBox.setWindowTitle("Invalid toml format")
        messageBox.setText("The written toml file is invalid.")
        messageBox.exec()



def getDisableEverythingFunction(window):
    def disableEverything(disable=True):
        global CURRENTLY_RUNNING 
        CURRENTLY_RUNNING = disable 
        
        window.filesMenu.setDisabled(disable)
        # window.loadFilesAction.setDisabled(disable)
        # window.saveFilesAction.setDisabled(disable)
        # window.saveSelectedMaskedImagesAction.setDisabled(disable)
        # window.saveMaskedImagesAction.setDisabled(disable)
        # window.saveSelectedFilesAction.setDisabled(disable)
        window.mainwidget.metadataWidget.runWidget.runAllButton.setDisabled(disable)
        window.mainwidget.metadataWidget.runWidget.runSelectedButton.setDisabled(disable)
        window.mainwidget.metadataWidget.runWidget.iceOrCarbonButton.setDisabled(disable)


    return disableEverything



def show_error_popup(etype, evalue,tb):
    QMessageBox.information(None, str('error'),''.join(format_exception(etype, evalue, tb)))





def main():
    # sys.excepthook = show_error_popup
    
    app = QApplication(sys.argv)
    window = MainWindow()
    
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()