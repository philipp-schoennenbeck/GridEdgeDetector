import glob
import math
import sys
from collections import namedtuple

from scipy import sparse
import subprocess, os, platform

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from PyQt5.QtCore import QAbstractTableModel, Qt, QSize, QThread, QObject, pyqtSignal, QModelIndex, QItemSelectionModel
from PyQt5.QtGui import QImage, QPixmap, QColor, QIntValidator, QDoubleValidator, QPen, QValidator
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView, QStyledItemDelegate, QWidget, QHBoxLayout, QVBoxLayout, QProgressBar, QPushButton, QGridLayout, QLabel, QLineEdit, QSizePolicy
from PyQt5.QtWidgets import QMenuBar, QMenu, QFileDialog, QFrame, QTabWidget, QPlainTextEdit
from pathlib import Path
import multiprocessing as mp
import mrcfile
import qimage2ndarray as q2n
import numpy as np
import psutil
from time import sleep
import random
import carbon_edge_detector as ced
from skimage.draw import disk
import toml
# from carbon_edge_detector
# import .carbon_edge_detector as ced



# Create a custom namedtuple class to hold our data.


Index = namedtuple("Index", ["row", "column"])
NUMBER_OF_COLUMNS = 7
CELL_PADDING = 5 # all sides
MAX_CORES = psutil.cpu_count()
CURRENTLY_RUNNING = False
DISABLE_FUNCTION = None
ALLOWED_SUFFIXES = set([".mrc", ".rec", ".MRC", ".REC"])
CONFIG_DIR = Path().home() / ".config" / "GridEdgeDetector"
if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir(parents=True)

CONFIG_FILE = CONFIG_DIR / "config.toml"
DEFAULT_CONFIG = {"title":"Grid edge detector configs", "parameters" :{"threshold":0.02, "gridsize":[2.0], "njobs":1}, "files":{"filedir":str(Path().home()), "addition to file":"_mask"}}

CURRENT_CONFIG = None

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
        toml.dump(DEFAULT_CONFIG,f)


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


def reevaluateMask(center, gridsize, ps, shape):
    yy,xx = disk(center, gridsize/ps // 2, shape=shape,)
    mask = np.zeros(shape, dtype=np.int8)
    
    mask[yy,xx] = 1
    return mask





def run_parallel(idx, fn, metadata, threshold):
    mask, hist_data, gridsize = ced.mask_carbon_edge_per_file(fn, [i * 10000 for i in metadata["Gridsize"]], threshold, metadata["Pixel spacing"], get_hist_data=True)
    return idx, mask, hist_data, gridsize

class Worker(QObject):

    finished = pyqtSignal()
    progress = pyqtSignal(tuple)
    def __init__(self, indexes, image_datas,threshold, njobs=1):
        super().__init__()
        self.njobs = njobs
        self.indexes = indexes
        self.image_datas = image_datas
        self.threshold = threshold

    def run(self):
        def callback(result):
            self.progress.emit(result)
            

        with mp.get_context("spawn").Pool(self.njobs) as pool:

            result = [pool.apply_async(run_parallel, [idx, data.fn, data.metadata, self.threshold], callback=callback) for idx, data in zip(self.indexes, self.image_datas)]
            [res.wait() for res in result]

        pool.join()
        
        
        

        # for i in range(5):
        #     sleep(1)
        #     self.progress.emit(i+1)
        #     self.progress.emit(self.njobs)
        self.finished.emit()
    





class image_data:
    def __init__(self, fn) -> None:
        global CURRENT_CONFIG
        self.fn = Path(fn)
        with mrcfile.open(fn,permissive=True) as f:

            data = f.data * 1
            self.metadata = {}
            self.metadata["Dimensions"] = data.shape
            self.metadata["Pixel spacing"] = float(f.voxel_size["x"])
            self.metadata["Gridsize"] = CURRENT_CONFIG["parameters"]["gridsize"]
        

        self.image  = q2n.gray2qimage(data, True)
        self.mask = None
        self.original_mask = None
        self.hist_data = None
        self.best_gridsize = None

    @property
    def title(self):
        return self.fn.name

class PreviewDelegate(QStyledItemDelegate):

    def paint(self, painter, option, index):
        # data is our preview object
        data = index.model().data(index, Qt.DisplayRole)
        if data is None:
            return

        width = option.rect.width() - CELL_PADDING * 2
        height = option.rect.height() - CELL_PADDING * 2

        # option.rect holds the area we are painting on the widget (our table cell)
        # scale our pixmap to fit
        scaled = data.image.scaled(
            width,
            height,
            aspectRatioMode=Qt.KeepAspectRatio,
        )
        # Position in the middle of the area.
        x = CELL_PADDING + (width - scaled.width()) / 2
        y = CELL_PADDING + (height - scaled.height()) / 2
        if data.mask is None:
            
            color = QColor("red")
            
        elif data.original_mask is not None and np.min(data.original_mask) == 0:
            color = QColor("green")
        else:
            color = QColor("yellow")
        painter.drawImage(option.rect.x() + x, option.rect.y() + y, scaled)
        painter.setPen(QPen(color, 3))
        painter.drawRect(option.rect.x() + x - 2, option.rect.y() + y - 2, scaled.width() + 4, scaled.height() + 4 )

    def sizeHint(self, option, index):
        # All items the same size.
        return QSize(100, 80)


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
            data = self.previews[index.row() * NUMBER_OF_COLUMNS + index.column() ]
            
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
        return NUMBER_OF_COLUMNS

    def rowCount(self, index=None):
        n_items = len(self.previews)
        return math.ceil(n_items / NUMBER_OF_COLUMNS)



class ImageViewer(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.view = QTableView()
        
        self.view.horizontalHeader().hide()
        self.view.verticalHeader().hide()
        self.view.setGridStyle(Qt.NoPen)

        delegate = PreviewDelegate()
        self.view.setItemDelegate(delegate)
        self.model = PreviewModel(self)
        self.view.setModel(self.model)
        
        self.view.setSelectionModel(customSelectionModel(self.model))
        self.view.selectionModel().selectionChanged.connect(self.selectionChanged)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.view)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed))

        self.setAcceptDrops(True)

    
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

            idxs = self.view.selectionModel().selectedIndexes()
            new_idxs = [idx.row() * self.model.columnCount() + idx.column() for idx in idxs]
            self.model.deleteIdxs(new_idxs)
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


    def load_files(self, files=None):

        for n, fn in enumerate(files):

            item = image_data(fn)
            self.model.previews.append(item)
            self.parent().setProgress((1+ n)/len(files) * 100)


        self.model.layoutChanged.emit()

        self.view.resizeRowsToContents()
        self.view.resizeColumnsToContents()









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
    
    def updatePlot(self, id):
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
            threshold = hist_data["threshold"]

            bargraph = pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=np.log(values),)
            
            line = pg.InfiniteLine(threshold,pen="red", movable=True)
            line.sigPositionChanged.connect(self.updateThreshold)
            line.sigPositionChangeFinished.connect(self.updateMask)
            current_plot_widget.setXRange(np.min(edges), max(np.max(edges), threshold))
            current_plot_widget.addItem(bargraph)
            current_plot_widget.addItem(line)
            # ax.bar(edges[:-1], values, width=np.diff(edges))

            self.plotItems[tab_name] = {"line":line, "bargraph":bargraph, "imagedata":id, "gridsize":key }
            self.plotTabWidget.addTab(current_plot_widget,tab_name)

    def clearAll(self):
        tab_count = self.plotTabWidget.count()
        for idx in range(tab_count-1, 0, -1):
            self.plotTabWidget.removeTab(idx)
        hist = self.plotTabWidget.widget(0)
        hist.clear()
        self.plotTabWidget.setTabText(0, "Hist")
        self.plotItems = {}

    def updateThreshold(self, line=None):

        line: pg.InfiniteLine

        self.parent().runWidget.thresholdLineEdit.setText(str(line.getPos()[0]))
        
    
    def updateMask(self, line=None):
        for key, value in self.plotItems.items():
            value["line"].setPos(line.getPos())
            
        pass
        if len(self.plotItems.keys()) != 0:
            newthreshold = line.getPos()[0]

            name = self.plotTabWidget.tabText(self.plotTabWidget.currentIndex())
            id = self.plotItems[name]["imagedata"]
            hist_data = id.hist_data[self.plotItems[name]["gridsize"]]
            edges = hist_data["edges"]
            values = hist_data["values"]
            self.plotTabWidget.currentWidget().setXRange(np.min(edges), max(np.max(edges), newthreshold))
            if np.max(edges) >= newthreshold:
                gridsize = self.plotItems[name]["gridsize"]
                new_mask = reevaluateMask(hist_data["center"], gridsize, id.metadata["Pixel spacing"],id.metadata["Dimensions"])

                
                # id.hist_data = hist_data
                
                
            else:
                gridsize = self.plotItems[name]["gridsize"]
                new_mask = np.zeros(id.metadata["Dimensions"], np.uint8)
            
            id.mask = q2n.array2qimage(new_mask, True)
            id.original_mask = new_mask
            id.best_gridsize = gridsize
            
            self.parent().thumbnailWidget.load_images(id)
            
            for idx in range(self.plotTabWidget.count()):
                widget = self.plotTabWidget.widget(idx)
                
                name = self.plotTabWidget.tabText(idx)
                id = self.plotItems[name]["imagedata"]
                hist_data = id.hist_data[id.best_gridsize]
                edges = hist_data["edges"]
                values = hist_data["values"]
            #     widget.setXRange(np.min(edges), max(np.max(edges), newthreshold))
                
    def sizeHint(self):
        # All items the same size.
        return QSize(350, 250)
        
            
    # def sizeHint(self):
    #     return QSize(240,240)

class thumbnailWidget(QFrame):
    shape = 120


    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Panel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.imageLabel = QLabel()
        self.maskLabel = QLabel()
        self.imagePixmap = QPixmap(self.shape,self.shape)
        self.maskPixmap = QPixmap(self.shape,self.shape)
        self.imagePixmap.fill(QColor("white"))
        self.maskPixmap.fill(QColor("white"))
        self.imageLabel.setPixmap(self.imagePixmap)
        self.maskLabel.setPixmap(self.maskPixmap)
        self.setLayout(QGridLayout())
        self.imageLabel.setStyleSheet("border: 1px solid black")
        self.maskLabel.setStyleSheet("border: 1px solid black")

        self.placeholder = QWidget()
        
        
        self.layout().addWidget(self.imageLabel,0,0)
        self.layout().addWidget(self.placeholder,0,1)
        self.layout().addWidget(self.maskLabel,0,2)
        self.layout().setColumnStretch(1,1)

    def load_images(self, data:image_data):
        image = data.image
        scaled = image.scaled(
            self.shape,
            self.shape,
            aspectRatioMode=Qt.KeepAspectRatio,
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
    
    def clearBoth(self):
        self.imagePixmap.fill(QColor("white"))
        self.maskPixmap.fill(QColor("white"))
        self.imageLabel.setPixmap(self.imagePixmap)
        self.maskLabel.setPixmap(self.maskPixmap)


class runWidget(QFrame):
    def __init__(self, parent):
        global CURRENT_CONFIG
        super().__init__(parent)
        self.setLayout(QGridLayout())

        # self.setStyleSheet("border: 1px solid black")
        # self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        # self.setLineWidth(1)
        self.setFrameShape(QFrame.Panel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.njobsLabel = QLabel("Njobs")
        self.njobsLineEdit = QLineEdit(str(CURRENT_CONFIG["parameters"]["njobs"]))
        self.thresholdLabel = QLabel("Threshold")
        self.thresholdLineEdit = QLineEdit(str(CURRENT_CONFIG["parameters"]["threshold"]))

        self.runAllButton = QPushButton(text="Mask all")
        self.runnSelectedButton = QPushButton(text="Mask selected")

        self.njobsLineEdit.setValidator(QIntValidator(1, MAX_CORES))
        self.thresholdLineEdit.setValidator(QDoubleValidator())
        self.runAllButton.clicked.connect(self.runAll)
        self.runnSelectedButton.clicked.connect(self.runSelected)
        self.number_of_images = 1
        self.current_number_of_images = 0
        self.layout().addWidget(self.njobsLabel, 0,0)
        self.layout().addWidget(self.njobsLineEdit, 0,1)
        self.layout().addWidget(self.thresholdLabel, 1,0)
        self.layout().addWidget(self.thresholdLineEdit, 1,1)
        self.layout().addWidget(self.runAllButton,2,0)
        self.layout().addWidget(self.runnSelectedButton,2,1)


    def runAll(self):
        if not self.currentlyRunning:
            global DISABLE_FUNCTION
            DISABLE_FUNCTION(True)              
            rows = self.parent().parent().imageviewer.model.rowCount(None)
            columns = self.parent().parent().imageviewer.model.columnCount(None)
            count = len(self.parent().parent().imageviewer.model.previews)

            indexes = [(row, col) for row in range(rows) for col in range(columns)]
            if count % columns != 0:
                indexes = indexes[:-(abs(count % columns - columns))]
            image_datas = self.parent().parent().imageviewer.model.previews
            self.runIndexes(indexes, image_datas)
            
            
            
    
    def runIndexes(self, indexes, image_datas):
        
        self.thread = QThread()
        self.worker = Worker(indexes, image_datas, float(self.thresholdLineEdit.text()), int(self.njobsLineEdit.text()))
        
        # self.worker.njobs = int(self.njobsLineEdit.text())
        self.number_of_images = len(indexes)
        self.current_number_of_images = 0
        self.parent().parent().setProgress(0)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.finishedRunning)
        self.worker.progress.connect(self.testWorker)
        self.thread.start()


    def testWorker(self, emit):
        (row, col), mask, hist_data, gridsize = emit
        self.current_number_of_images += 1
        self.parent().parent().setProgress(self.current_number_of_images / self.number_of_images * 100)
        view:QTableView = self.parent().parent().imageviewer.view 
        data:image_data = view.model().index(row,col).data()
        data.mask = q2n.array2qimage(mask, True)
        data.original_mask = mask
        data.hist_data = hist_data
        data.best_gridsize = gridsize
        index = view.model().createIndex(row, col)
        view.selectionModel().select(index,QItemSelectionModel.SelectionFlag.ClearAndSelect)
        view.model().dataChanged.emit(index, index)

    def finishedRunning(self):
        global DISABLE_FUNCTION
        self.parent().parent().setProgress(100)
        
        DISABLE_FUNCTION(False)

    def runSelected(self):
        if not self.currentlyRunning:
            global DISABLE_FUNCTION
            DISABLE_FUNCTION(True)
            idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
            column_count = self.parent().parent().imageviewer.model.columnCount()
            
            new_idxs = [idx.row() * column_count + idx.column() for idx in idxs]

            image_datas = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
            idxs = [(idx.row(), idx.column()) for idx in idxs]
            self.runIndexes(idxs, image_datas)

    @property
    def currentlyRunning(self):
        global CURRENTLY_RUNNING
        return CURRENTLY_RUNNING
    
    @currentlyRunning.setter
    def currentlyRunning(self, value):
        global CURRENTLY_RUNNING
        CURRENTLY_RUNNING = value


class rightWidget(QWidget):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.histWidget = histWidget(self)
        self.thumbnailWidget = thumbnailWidget(self)
        self.metadataWidget = metadataWidget(self)
        self.runWidget = runWidget(self)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.histWidget)
        self.layout().addWidget(self.thumbnailWidget)
        self.layout().addWidget(self.metadataWidget)
        self.layout().addWidget(self.runWidget)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed))

    def show_data(self, idxs):
        data = self.metadataWidget.show_data(idxs)
        if len(idxs) == 1:
            assert len(data) == 1
            data = data[0]
            self.thumbnailWidget.load_images(data)
            self.histWidget.updatePlot(data)
        else:
            self.thumbnailWidget.clearBoth()
            self.histWidget.clearAll()

    def sizeHint(self):
        # All items the same size.
        return QSize(350, 600)


class QFloatListValidator(QValidator):
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

        return super().validate(a0, a1)





class metadataWidget(QFrame):
    def __init__(self, parent):
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
        

        self.pixelspacingLabelDesc = QLabel("Pixel spacing [px/Å]:")
        self.pixelspacingLabel = QLabel("")

        self.gridsizeLabelDesc = QLabel("Grid hole size [µm]:")
        self.gridsizeLineEdit = QLineEdit("")

        self.gridsizeLineEdit.setValidator(QFloatListValidator())
        self.gridsizeLineEdit.setToolTip("Size of the grid hole sizes in µm. If unsure, you can input multiple values seperates by commas and it will try to find the best one.")
        self.gridsizeLineEdit.editingFinished.connect(self.setGridsize)

        # self.layout().setVerticalSpacing(0)
        # (self.fileNameLabelDesc, self.fileNameLabel)
        for counter, (desc, label) in enumerate([(self.dimensionLabelDesc, self.dimensionLabel),(self.pixelspacingLabelDesc, self.pixelspacingLabel),(self.gridsizeLabelDesc, self.gridsizeLineEdit)]):

            self.layout().addWidget(desc, counter, 0)
            self.layout().addWidget(label, counter, 1)
            if counter == 2:
                color = "white"
            else:
                color = "lightgray"
            label.setStyleSheet(f"border: 1px solid black gray;background-color: {color}")

        self.layout().setRowStretch(counter + 1, 1)
        self.layout().setColumnStretch(1,1)
        
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
        self.dimensionLabel.setText(", ".join(dimensions))
        self.pixelspacingLabel.setText(", ".join(pixespacings))
        self.gridsizeLineEdit.setText(", ".join(gridsizes))
        return ids
        # else:
        #     self.dimensionLabel.setText(str(data.metadata["Dimensions"]))
        #     self.pixelspacingLabel.setText(str(data.metadata["Pixel spacing"]))
        #     self.gridsizeLineEdit.setText(", ".join([str(i) for i in data.metadata["Gridsize"]]))

    def setGridsize(self, event=None):
        gridsizes = self.gridsizeLineEdit.text().replace(" ", "").split(",")
        new_gridsizes = [float(gs) for gs in gridsizes if len(gs) > 0]

        idxs = self.parent().parent().imageviewer.view.selectionModel().selectedIndexes()
        new_idxs = [idx.row() * self.parent().parent().imageviewer.model.columnCount() + idx.column() for idx in idxs]
        ids = [self.parent().parent().imageviewer.model.previews[idx] for idx in new_idxs]
        for id in ids:
            id.metadata["Gridsize"] = new_gridsizes


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


    def pushed_button(self):
        self.imageviewer.load_files()

    def setProgress(self, progress):
        progress = round(progress)
        self.progressBar.setValue(progress)


    def sizeHint(self):
        # All items the same size.
        return QSize(1600, 1000)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.mainwidget = mainWidget(self)
        self.createMenu()
        self.setCentralWidget(self.mainwidget)
        self.setWindowTitle("Grid edge detector")
    
    def createMenu(self):
        menuBar = self.menuBar()
        self.filesMenu = QMenu("Files", menuBar)
        self.loadFilesAction = self.filesMenu.addAction("Load files")
        self.saveFilesAction = self.filesMenu.addAction("Save all masks")
        self.saveSelectedFilesAction = self.filesMenu.addAction("Save selected masks")
        menuBar.addMenu(self.filesMenu)
        
        self.loadFilesAction.triggered.connect(self.loadFiles)
        self.saveFilesAction.triggered.connect(self.saveAllMasks)
        self.saveSelectedFilesAction.triggered.connect(self.saveSelectedMasks)

        
        self.configMenu = QMenu("Config", menuBar)
        self.openConfigFileAction = self.configMenu.addAction("Open config file")
        self.createDefaultConfigFileAction = self.configMenu.addAction("Create default config file")
        self.openConfigFileAction.triggered.connect(self.openConfigFile)
        self.createDefaultConfigFileAction.triggered.connect(self.createDefaultConfig)
        menuBar.addMenu(self.configMenu)

    def openConfigFile(self):
        global CONFIG_FILE, CURRENT_CONFIG
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', CONFIG_FILE))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(CONFIG_FILE)
        else:                                   # linux variants
            subprocess.call(('xdg-open', CONFIG_FILE))
        
        CURRENT_CONFIG = load_config_file()
        self.loadConfig()
    
    def loadConfig(self):
        global CURRENT_CONFIG
        previews = self.mainwidget.imageviewer.model.previews
        for i in previews:
            i:image_data 
            i.metadata["Gridsize"] = CURRENT_CONFIG["parameters"]["gridsize"]
        self.mainwidget.metadataWidget.runWidget.njobsLineEdit.setText(str(CURRENT_CONFIG["parameters"]["njobs"]))
        self.mainwidget.metadataWidget.runWidget.thresholdLineEdit.setText(str(CURRENT_CONFIG["parameters"]["threshold"]))

    def createDefaultConfig(self):
        global CURRENT_CONFIG
        create_default_config_file(True)
        CURRENT_CONFIG = load_config_file()
        self.loadConfig()


    def setDisabledActions(self, disable=True):
        global DISABLE_FUNCTION
        DISABLE_FUNCTION(disable)
        # self.loadFilesAction.setDisabled(disable)
        # self.saveFilesAction.setDisabled(disable)
        # self.saveSelectedFilesAction.setDisabled(disable)
        # self.mainwidget.metadataWidget.runWidget.runAllButton.setDisabled(disable)
        # self.mainwidget.metadataWidget.runWidget.runnSelectedButton.setDisabled(disable)

    def loadFiles(self):
        global DISABLE_FUNCTION
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.ExistingFiles)
        names, filt = dialog.getOpenFileNames(self, "Open images", CURRENT_CONFIG["files"]["filedir"], "mrc files (*.mrc *.MRC *.rec, *.REC)")
        # dialog.getOpenFileNames(self, "Open images", "/home", )
        # filedialog = QFileDialog.getOpenFileName(self, "Open Image", "/home", "")
        names = [Path(name) for name in names]
        DISABLE_FUNCTION(True)
        self.mainwidget.imageviewer.load_files(names)
        DISABLE_FUNCTION(False)

    def saveAllMasks(self):
        image_datas = self.mainwidget.imageviewer.model.previews
        self.saveMasks(image_datas)
    
    def saveSelectedMasks(self):
        pass

    def saveMasks(self, image_datas):
        global CURRENT_CONFIG
        
        dialog = QFileDialog()
        # dialog.setFileMode(QFileDialog.Ex)
        save_dir = dialog.getExistingDirectory(self, "Save directory", CURRENT_CONFIG["files"]["filedir"])
        if save_dir is not None and len(save_dir) > 0:
            save_dir = Path(save_dir).absolute()
            for id in image_datas:
                if id.original_mask is not None:
                    current_path = save_dir / (id.fn.stem + CURRENT_CONFIG["files"]["addition to file"] + id.fn.suffix)
                    with mrcfile.new(current_path, data=id.original_mask, overwrite=True) as f:
                        f.voxel_size = id.metadata["Pixel spacing"]
        




def getDisableEverythingFunction(window):
    def disableEverything(disable=True):
        global CURRENTLY_RUNNING 
        CURRENTLY_RUNNING = disable 

        window.loadFilesAction.setDisabled(disable)
        window.saveFilesAction.setDisabled(disable)
        window.saveSelectedFilesAction.setDisabled(disable)
        window.mainwidget.metadataWidget.runWidget.runAllButton.setDisabled(disable)
        window.mainwidget.metadataWidget.runWidget.runnSelectedButton.setDisabled(disable)

    return disableEverything


if __name__ == "__main__":
    CURRENT_CONFIG = load_config_file()
    
    app = QApplication(sys.argv)
    window = MainWindow()
    DISABLE_FUNCTION = getDisableEverythingFunction(window)
    window.show()
    app.exec_()