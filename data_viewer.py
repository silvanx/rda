import pickle
import re
import sys

import matplotlib
from PyQt5 import QtWidgets, QtCore, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,\
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pathlib import Path
import numpy as np
import scipy.signal as signal
from ratdata import ingest, data_manager as dm

matplotlib.use('Qt5Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        dm.db_connect('rat_data.db')

        self.file_dir = 'data/mce_recordings'
        self.current_file = None
        self.rat_selected = None
        self.condition_selected = None
        self.stim_selected = None
        self.file_list = self.populate_file_list()

        self.all_conditions = [None, 'baseline', 'ST', 'CT', 'OFT']
        # self.all_rats = [None, 'rat1', 'rat2', 'rat3', 'rat5']
        self.all_rats = dm.get_rat_labels()
        self.all_stims = [None, 'nostim', 'continuous', 'on-off',
                          'random', 'proportional']

        self.time_slices_file = Path(self.file_dir) / 'time_slices.pickle'
        if self.time_slices_file.exists():
            with open(self.time_slices_file, 'rb') as f:
                self.time_slices = pickle.load(f)
        else:
            self.time_slices = dict()

        self.time_plot = MplCanvas(self, width=5, height=4, dpi=100)
        self.psd_plot = MplCanvas(self, width=5, height=3, dpi=100)

        # Create matplotlib edit toolbar
        toolbar = NavigationToolbar(self.time_plot, self)
        toolbar_psd = NavigationToolbar(self.psd_plot, self)

        # Create next/previous buttons
        nav_buttons = QtWidgets.QHBoxLayout()
        prev_button = QtWidgets.QPushButton("< Previous")
        next_button = QtWidgets.QPushButton("Next >")
        nav_buttons.addWidget(prev_button)
        nav_buttons.addWidget(next_button)

        # Select slice controls
        slice_selection = QtWidgets.QHBoxLayout()
        self.start_slice_input = QtWidgets.QLineEdit()
        self.length_slice_input = QtWidgets.QLineEdit()
        update_slice_button = QtWidgets.QPushButton("Update selection")
        slice_selection.addWidget(QtWidgets.QLabel("Slice start:"))
        slice_selection.addWidget(self.start_slice_input)
        slice_selection.addWidget(QtWidgets.QLabel("Slice length:"))
        slice_selection.addWidget(self.length_slice_input)
        slice_selection.addWidget(update_slice_button)
        update_slice_button.clicked.connect(self.update_selected_slice)
        self.start_slice_input.returnPressed.connect(
            self.update_selected_slice)
        self.length_slice_input.returnPressed.connect(
            self.update_selected_slice)

        # Time plot controls
        time_plot_controls = QtWidgets.QHBoxLayout()
        time_plot_controls.addWidget(toolbar)
        time_plot_controls.addLayout(slice_selection)

        # Plotting areas
        plot_area = QtWidgets.QVBoxLayout()
        plot_area.addLayout(time_plot_controls)
        plot_area.addWidget(self.time_plot)
        plot_area.addWidget(toolbar_psd)
        plot_area.addWidget(self.psd_plot)
        plot_area.addLayout(nav_buttons)

        # File list on the right
        self.file_list_widget = QtWidgets.QListWidget()
        self.directory_label = QtWidgets.QLabel(self.file_dir)
        change_directory_button = QtWidgets.QPushButton('Change directory')
        self.refresh_file_list_display()

        condition_combo_box = QtWidgets.QComboBox()
        condition_combo_box.insertItems(0, self.all_conditions)
        condition_combo_box_area = QtWidgets.QHBoxLayout()
        condition_combo_box_area.addWidget(QtWidgets.QLabel('Condition: '))
        condition_combo_box_area.addWidget(condition_combo_box)
        condition_combo_box.currentIndexChanged.connect(
            self.filter_files_by_condition)

        rat_combo_box = QtWidgets.QComboBox()
        rat_combo_box.insertItems(0, self.all_rats)
        rat_combo_box_area = QtWidgets.QHBoxLayout()
        rat_combo_box_area.addWidget(QtWidgets.QLabel('Rat: '))
        rat_combo_box_area.addWidget(rat_combo_box)
        rat_combo_box.currentIndexChanged.connect(self.filter_files_by_rat)

        stim_combo_box = QtWidgets.QComboBox()
        stim_combo_box.insertItems(0, self.all_stims)
        stim_combo_box_area = QtWidgets.QHBoxLayout()
        stim_combo_box_area.addWidget(QtWidgets.QLabel('Stim: '))
        stim_combo_box_area.addWidget(stim_combo_box)
        stim_combo_box.currentIndexChanged.connect(self.filter_files_by_stim)

        file_list_area = QtWidgets.QVBoxLayout()
        file_list_area.addWidget(QtWidgets.QLabel('Current directory:'))
        file_list_area.addWidget(self.directory_label)
        file_list_area.addWidget(change_directory_button)
        file_list_area.addLayout(condition_combo_box_area)
        file_list_area.addLayout(rat_combo_box_area)
        file_list_area.addLayout(stim_combo_box_area)
        file_list_area.addWidget(self.file_list_widget)

        layout = QtWidgets.QHBoxLayout()
        layout.addLayout(plot_area, stretch=4)
        layout.addLayout(file_list_area, stretch=1)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        next_button.clicked.connect(self.plot_next_file)
        prev_button.clicked.connect(self.plot_previous_file)
        change_directory_button.clicked.connect(self.change_file_dir)

        self.file_list_widget.itemSelectionChanged.connect(
            self.plot_clicked_file)

        self.setCentralWidget(widget)
        self.showMaximized()

        self.show()

    def change_file_dir(self):
        newdir = QtWidgets.QFileDialog.getExistingDirectory()
        self.file_dir = newdir
        self.directory_label.setText(newdir)
        self.current_file = None
        self.file_list = self.populate_file_list()
        self.refresh_file_list_display()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_PageDown:
            self.plot_next_file()
        if event.key() == QtCore.Qt.Key_PageUp:
            self.plot_previous_file()

    def refresh_file_list_display(self):
        self.file_list_widget.clearSelection()
        self.file_list_widget.clear()
        for file in self.file_list:
            currentItem = QtWidgets.QListWidgetItem(self.file_list_widget)
            currentItem.setText(file)
            if file.split('.')[0] in self.time_slices:
                currentItem.setIcon(QtGui.QIcon(r"scissors.png"))

    def plot_clicked_file(self):
        if len(self.file_list_widget.selectedItems()) > 0:
            item = self.file_list_widget.selectedItems()[0]
            text = item.text()
            self.current_file = self.file_list.index(text)
            filename = self.file_dir + '/' + self.file_list[self.current_file]
            self.plot_data_from_file(filename)

    def plot_next_file(self):
        if self.current_file is None:
            self.current_file = 0
        else:
            self.current_file = (self.current_file + 1) % len(self.file_list)
        filename = self.file_dir + '/' + self.file_list[self.current_file]
        self.plot_data_from_file(filename)
        self.file_list_widget.setCurrentRow(self.current_file)

    def plot_previous_file(self):
        if self.current_file is None:
            self.current_file = 0
        else:
            self.current_file = (self.current_file - 1) % len(self.file_list)
        filename = self.file_dir + '/' + self.file_list[self.current_file]
        self.plot_data_from_file(filename)
        self.file_list_widget.setCurrentRow(self.current_file)

    def plot_data_from_file(self, filename):
        file = Path(filename)
        self.start_slice_input.clear()
        self.length_slice_input.clear()
        if re.match(r'^\.mat$', file.suffix):
            self.plot_data_from_matlab_file(filename)
        elif re.match(r'^\.txt$', file.suffix):
            self.plot_data_from_gui_csv(filename)
        elif re.match(r'^\.bin$', file.suffix):
            self.plot_data_from_gui_bin(filename)

    def plot_data_from_matlab_file(self, filename):
        file = Path(filename)
        if re.match(r'^\.mat$', file.suffix):
            data = ingest.read_mce_matlab_file(filename)
            samples = data.electrode_data.shape[1]
            tt = np.linspace(0, samples * data.dt, samples)
            x = np.mean(data.electrode_data, 0)
            self.update_plot(tt, x, file)

    def plot_data_from_gui_csv(self, filename):
        pass
        # file = Path(filename)
        # # TODO: Save and read metadata, including sampling frequency
        # fs = 20000
        # if re.match(r'^\.txt$', file.suffix):
        #     data = rattools.load_gui_csv_recording(file)
        #     samples = data.shape[0]
        #     tt = np.linspace(0, samples / fs, samples)
        #     x = np.mean(data[:, 0:4], 1)
        #     self.update_plot(tt, x, file)

    def plot_data_from_gui_bin(self, filename):
        pass
        # file = Path(filename)
        # # TODO: Save and read metadata, including sampling frequency
        # fs = 20000
        # if re.match('^\.bin$', file.suffix):
        #     data = rattools.load_gui_bin_recording(file)
        #     samples = data.shape[0]
        #     tt = np.linspace(0, samples / fs, samples)
        #     x = np.mean(data[:, 0:4], 1)
        #     self.update_plot(tt, x, file)

    def update_plot(self, tt, x, file):
        self.time_plot.axes.cla()
        self.time_plot.axes.plot(tt, x)
        self.time_plot.axes.set_xlabel('Time [s]')
        self.time_plot.axes.set_title(file.stem)
        if file.stem in self.time_slices:
            elem = self.time_slices[file.stem]
            self.start_slice_input.clear()
            self.start_slice_input.insert(str(elem['start']))
            self.length_slice_input.clear()
            self.length_slice_input.insert(str(elem['length']))
            self.time_plot.axes.fill_betweenx([min(x), max(x)],
                                              elem['start'],
                                              elem['start'] + elem['length'],
                                              color='green', alpha=0.2)
        self.time_plot.fig.tight_layout()
        self.time_plot.draw()

        fs = 20000
        if file.stem in self.time_slices:
            elem = self.time_slices[file.stem]
            start_sample = int(elem['start'] * fs)
            end_sample = int((elem['start'] + elem['length']) * fs)
            x = x[start_sample:end_sample]
        f_signal, spectrum_signal = signal.welch(x, fs, nperseg=fs)
        self.psd_plot.axes.cla()
        self.psd_plot.axes.plot(f_signal, spectrum_signal)
        self.psd_plot.axes.set_xlim([0, 150])
        max_150 = max(spectrum_signal[f_signal <= 150]) * 1.05
        self.psd_plot.axes.set_ylim([0, max_150])
        self.psd_plot.axes.set_xlabel('Frequency [Hz]')
        self.psd_plot.axes.set_title('PSD')

        self.psd_plot.fig.tight_layout()
        self.psd_plot.draw()

    def populate_file_list(self):
        dir = Path(self.file_dir)
        extension_regexp = re.compile(r'.*\.(mat|txt|bin)$')
        stim_regexp_dict = {
            'nostim': r'.* (baseline[0-9]?|CT( 90s)?|ST|OFT[12]?)\.mat$',
            'continuous': r'.* (130Hz|DBS)\.mat$',
            'on-off': r'.*on-off\.mat$',
            'random': r'.*random\.mat$',
            'proportional': r'.*pro( noscale)?\.mat$'
        }
        if self.rat_selected is not None:
            rat_regexp = re.compile(
                r'.+ %s .*\.(mat|txt|bin)$' % self.rat_selected)
        else:
            rat_regexp = extension_regexp
        if self.condition_selected is not None:
            condition_regexp = re.compile(
                r'.+ %s.*\.(mat|txt|bin)$' % self.condition_selected)
        else:
            condition_regexp = extension_regexp
        if self.stim_selected is not None:
            stim_regexp = re.compile(stim_regexp_dict[self.stim_selected])
        else:
            stim_regexp = extension_regexp

        file_list = [file.name for file in dir.iterdir() if
                     re.match(rat_regexp, file.name) and
                     re.match(condition_regexp, file.name) and
                     re.match(stim_regexp, file.name)]
        return file_list

    def filter_files_by_condition(self, index):
        self.condition_selected = self.all_conditions[index]
        self.file_list = self.populate_file_list()
        self.refresh_file_list_display()

    def filter_files_by_rat(self, index):
        self.rat_selected = self.all_rats[index]
        self.file_list = self.populate_file_list()
        self.refresh_file_list_display()

    def filter_files_by_stim(self, index):
        self.stim_selected = self.all_stims[index]
        self.file_list = self.populate_file_list()
        self.refresh_file_list_display()

    def update_selected_slice(self):
        filename = Path(self.file_list[self.current_file])
        start = self.start_slice_input.text()
        length = self.length_slice_input.text()
        if start and length:
            self.time_slices[filename.stem] = {'start': float(start),
                                               'length': float(length)}
            self.file_list_widget.selectedItems()[0].setIcon(
                QtGui.QIcon(r"scissors.png"))
        else:
            self.time_slices.pop(filename.stem)
            self.file_list_widget.selectedItems()[0].setIcon(QtGui.QIcon(None))
        with open(self.time_slices_file, mode='wb') as f:
            pickle.dump(self.time_slices, f)
        full_filename = Path(self.file_dir) / filename.name
        self.plot_data_from_file(full_filename)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
