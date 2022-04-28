import pathlib
import pickle
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,\
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from pathlib import Path
import numpy as np
import scipy.signal as signal
from ratdata import ingest, data_manager as dm, process

matplotlib.use('Qt5Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class PulseTemplate:

    def __init__(self, length: int, channels: int, template: np.ndarray,
                 start: list[int], align: str) -> None:
        self.length: int = length
        self.channels: int = channels
        self.template: np.ndarray = template
        self.start: list[int] = start
        self.align: str = align


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.file_dir = 'data/mce_recordings'
        self.time_slices_file = Path(self.file_dir) / 'time_slices.pickle'
        self.db_file = 'rat_data.db'

        dm.db_connect(self.db_file)

        self.current_file = None
        self.rat_selected = None
        self.condition_selected = None
        self.stim_selected = None

        self.oof_psd_visible = False
        self.oof_params_updated = False
        self.oof_equation = None

        self.welch_window_length = 1

        self.file_list = self.populate_file_list()

        self.all_conditions = dm.get_condition_labels()
        self.all_rats = dm.get_rat_labels()
        self.all_stims = dm.get_stim_types()
        # self.all_rats = [None, 'rat1', 'rat2', 'rat3', 'rat5']
        # self.all_conditions = [None, 'baseline', 'ST', 'CT', 'OFT']
        # self.all_stims = [None, 'nostim', 'continuous', 'on-off',
        #                   'random', 'proportional']

        self.time_slices = ingest.read_file_slices(self.time_slices_file)

        self.time_plot = MplCanvas(self, width=5, height=4, dpi=100)
        self.psd_plot = MplCanvas(self, width=5, height=3, dpi=100)

        self.pulse_template = None
        self.subtract_pulse = False

        # Matplotlib edit toolbar
        toolbar = NavigationToolbar(self.time_plot, self)
        toolbar_psd = NavigationToolbar(self.psd_plot, self)

        # Next/previous buttons
        nav_buttons = QtWidgets.QHBoxLayout()
        prev_button = QtWidgets.QPushButton("< Previous")
        next_button = QtWidgets.QPushButton("Next >")
        nav_buttons.addWidget(prev_button)
        nav_buttons.addWidget(next_button)

        show_stim_spikes_button = QtWidgets.QPushButton("Stim pulse")
        self.stim_pulse_window = StimPulseWindow(self)
        show_stim_spikes_button.clicked.connect(self.show_stim_pulse_window)

        # Select slice controls
        slice_selection = QtWidgets.QHBoxLayout()
        self.start_slice_input = QtWidgets.QLineEdit()
        self.length_slice_input = QtWidgets.QLineEdit()
        update_slice_button = QtWidgets.QPushButton("Update selection")
        reject_file_button = QtWidgets.QPushButton("Reject file")
        slice_selection.addWidget(QtWidgets.QLabel("Slice start:"))
        slice_selection.addWidget(self.start_slice_input)
        slice_selection.addWidget(QtWidgets.QLabel("Slice length:"))
        slice_selection.addWidget(self.length_slice_input)
        slice_selection.addWidget(update_slice_button)
        slice_selection.addWidget(reject_file_button)
        slice_selection.addWidget(show_stim_spikes_button)
        update_slice_button.clicked.connect(self.update_selected_slice)
        reject_file_button.clicked.connect(self.reject_file)
        self.start_slice_input.returnPressed.connect(
            self.update_selected_slice)
        self.length_slice_input.returnPressed.connect(
            self.update_selected_slice)

        # Time plot controls
        time_plot_controls = QtWidgets.QHBoxLayout()
        time_plot_controls.addWidget(toolbar)
        time_plot_controls.addLayout(slice_selection)

        self.welch_length_input = QtWidgets.QLineEdit()
        self.welch_length_input.setText(str(self.welch_window_length))
        self.welch_length_input.editingFinished.connect(self.update_psd)

        # one-over-f component in PSD
        psd_oof = QtWidgets.QHBoxLayout()
        self.psd_oof_show = QtWidgets.QPushButton('Show')
        self.psd_oof_show.clicked.connect(self.toggle_oof_psd)
        self.oof_freq_low = QtWidgets.QLineEdit()
        self.oof_freq_low.setText('2')
        self.oof_freq_low.textChanged.connect(self.update_oof_params)
        self.oof_freq_high = QtWidgets.QLineEdit()
        self.oof_freq_high.setText('100')
        self.oof_freq_high.textChanged.connect(self.update_oof_params)
        self.oof_scale = QtWidgets.QLineEdit()
        self.oof_scale.setText('1')
        self.oof_scale.textChanged.connect(self.update_oof_params)
        psd_oof.addWidget(QtWidgets.QLabel('1/f fitting: '))
        psd_oof.addWidget(QtWidgets.QLabel('Low frequency:'))
        psd_oof.addWidget(self.oof_freq_low)
        psd_oof.addWidget(QtWidgets.QLabel('High frequency:'))
        psd_oof.addWidget(self.oof_freq_high)
        psd_oof.addWidget(QtWidgets.QLabel('Scale:'))
        psd_oof.addWidget(self.oof_scale)
        psd_oof.addWidget(self.psd_oof_show)

        psd_plot_controls = QtWidgets.QHBoxLayout()
        psd_plot_controls.addWidget(toolbar_psd)
        psd_plot_controls.addWidget(QtWidgets.QLabel(
            'Welch window length (s):'))
        psd_plot_controls.addWidget(self.welch_length_input)
        psd_plot_controls.addLayout(psd_oof)

        # Plotting areas
        plot_area = QtWidgets.QVBoxLayout()
        plot_area.addLayout(time_plot_controls)
        plot_area.addWidget(self.time_plot)
        plot_area.addLayout(psd_plot_controls)
        plot_area.addWidget(self.psd_plot)
        plot_area.addLayout(nav_buttons)

        # File list on the right
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.setIconSize(QtCore.QSize(12, 12))
        self.db_label = QtWidgets.QLabel(self.db_file)
        self.db_label.setStyleSheet("QLabel::hover"
                                    "{"
                                    "color : #8e8e8e"
                                    "}")
        self.directory_label = QtWidgets.QLabel(self.file_dir)
        self.directory_label.setStyleSheet("QLabel::hover"
                                           "{"
                                           "color : #8e8e8e"
                                           "}")
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
        file_list_area.addWidget(QtWidgets.QLabel('Current database:'))
        file_list_area.addWidget(self.db_label)
        file_list_area.addWidget(QtWidgets.QLabel('Current directory:'))
        file_list_area.addWidget(self.directory_label)
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
        self.directory_label.mousePressEvent = self.change_file_dir

        self.file_list_widget.itemSelectionChanged.connect(
            self.plot_clicked_file)

        self.refresh_file_list_display()
        self.setWindowTitle('Rat data analysis')
        self.setCentralWidget(widget)
        self.showMaximized()

        self.show()

    def error_box(self, text: str):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText('Error')
        msg.setInformativeText(text)
        msg.setWindowTitle('Error')
        msg.exec()

    def update_oof_params(self):
        if self.oof_psd_visible:
            self.oof_params_updated = True
            self.psd_oof_show.setText('Update')

    def toggle_oof_psd(self):
        error = ''
        try:
            _ = float(self.oof_freq_low.text())
        except ValueError:
            error = 'Min frequency must be a number\n'
        try:
            _ = float(self.oof_freq_high.text())
        except ValueError:
            error += 'Max frequency must be a number\n'
        try:
            scale = float(self.oof_scale.text())
            if scale < 0:
                raise ValueError
        except ValueError:
            error += 'Scale parameter must be a positive number'

        if error != '':
            self.error_box(error)
        else:
            if self.oof_params_updated:
                self.hide_oof_psd()
                self.oof_params_updated = False
                self.oof_psd_visible = False
            self.oof_psd_visible = not self.oof_psd_visible
            if self.oof_psd_visible:
                self.show_oof_psd()
                self.psd_oof_show.setText('Hide')
            else:
                self.hide_oof_psd()
                self.psd_oof_show.setText('Show')

    def hide_oof_psd(self):
        ax = self.psd_plot.axes
        n_plots = len(ax.lines)
        if n_plots > 1:
            for i in range(1, n_plots):
                ax.lines[-1].remove()
            ax.get_legend().remove()
            self.psd_plot.draw()

    def show_oof_psd(self):
        ax = self.psd_plot.axes
        f_min = float(self.oof_freq_low.text())
        f_max = float(self.oof_freq_high.text())
        scale = float(self.oof_scale.text())
        if len(ax.lines) > 0:
            current_psd_line = ax.lines[0]
            f = current_psd_line._x
            pxx = current_psd_line._y
            m, b = process.fit_oof(f, pxx, f_min, f_max)
            f[f == 0] = 0.000000001
            fm = f ** m
            oof = scale * (np.e**b * fm)
            clean_pxx = pxx - oof
            equation = '$e^{%.2f} \\times f^{%.2f}$' % (b, m)
            ax.plot(current_psd_line._x, oof)
            color = self.psd_plot.axes.lines[-1].get_color()
            ax.text(75, ax.get_ylim()[1] / 2, equation,
                    color=color)
            ax.plot(current_psd_line._x, clean_pxx)
            ax.legend(['PSD', '1/f', 'PSD-1/f'])
            self.psd_plot.draw()

    def change_file_dir(self, event):
        newdir = QtWidgets.QFileDialog.getExistingDirectory(self, '',
                                                            self.file_dir)
        if newdir:
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
            if dm.is_recording_rejected(file):
                currentItem.setIcon(QtGui.QIcon(r'stop.png'))
            elif dm.is_recording_sliced(file):
                currentItem.setIcon(QtGui.QIcon(r'scissors.png'))
            else:
                currentItem.setIcon(QtGui.QIcon(r'empty.png'))
            # file_key = file.split('.')[0]
            # if file_key in self.time_slices:
            #     # if 'reject' in self.time_slices[file_key]:
            #     #     currentItem.setIcon(QtGui.QIcon(r"stop.png"))
            #     # else:
            #     currentItem.setIcon(QtGui.QIcon(r"scissors.png"))

    def plot_clicked_file(self):
        if len(self.file_list_widget.selectedItems()) > 0:
            item = self.file_list_widget.selectedItems()[0]
            text = item.text()
            self.current_file = self.file_list.index(text)
            filename = self.file_dir + '/' + self.file_list[self.current_file]
            if self.stim_pulse_window.start_markers is not None:
                self.stim_pulse_window.toggle_start_on_main_plot()
            if self.stim_pulse_window.end_markers is not None:
                self.stim_pulse_window.toggle_end_on_main_plot()
            self.subtract_pulse = False
            self.plot_data_from_file(filename)

    def update_psd(self):
        new_welch_len = int(self.welch_length_input.text())
        if self.welch_window_length != new_welch_len:
            self.welch_window_length = new_welch_len
            current_filename = self.file_list[self.current_file]
            if current_filename:
                self.plot_data_from_file(
                    Path(self.file_dir) / current_filename)

    def plot_next_file(self):
        if self.current_file is None:
            self.current_file = 0
        else:
            self.current_file = (self.current_file + 1) % len(self.file_list)
        filename = self.file_dir + '/' + self.file_list[self.current_file]
        if self.stim_pulse_window.start_markers is not None:
            self.stim_pulse_window.toggle_start_on_main_plot()
        if self.stim_pulse_window.end_markers is not None:
            self.stim_pulse_window.toggle_end_on_main_plot()
        self.subtract_pulse = False
        self.plot_data_from_file(filename)
        self.file_list_widget.setCurrentRow(self.current_file)

    def plot_previous_file(self):
        if self.current_file is None:
            self.current_file = 0
        else:
            self.current_file = (self.current_file - 1) % len(self.file_list)
        filename = self.file_dir + '/' + self.file_list[self.current_file]
        if self.stim_pulse_window.start_markers is not None:
            self.stim_pulse_window.toggle_start_on_main_plot()
        if self.stim_pulse_window.end_markers is not None:
            self.stim_pulse_window.toggle_end_on_main_plot()
        self.subtract_pulse = False
        self.plot_data_from_file(filename)
        self.file_list_widget.setCurrentRow(self.current_file)

    def plot_data_from_file(self, filename):
        file = Path(filename)
        self.start_slice_input.clear()
        self.length_slice_input.clear()
        if re.match(r'^\.mat$', file.suffix):
            self.plot_data_from_matlab_file(filename)
        elif re.match(r'^\.txt$', file.suffix):
            self.plot_data_from_amplitude_file(filename)
        elif re.match(r'^\.bin$', file.suffix):
            self.plot_data_from_gui_bin(filename)
        if self.oof_psd_visible:
            self.show_oof_psd()

    def plot_data_from_matlab_file(self, filename):
        file = Path(filename)
        if re.match(r'^\.mat$', file.suffix):
            data = ingest.read_mce_matlab_file(filename)
            data.slice = dm.get_recording_slice(file.name)
            samples = data.electrode_data.shape[1]
            tt = np.linspace(0, samples * data.dt, samples)
            if self.subtract_pulse:
                x = process.subtract_template(data.electrode_data,
                                              self.pulse_template)
            else:
                x = data.electrode_data
            if len(x.shape) > 1:
                x = np.mean(x, 0)
            self.stim_pulse_window.set_recording(data)
            self.update_plot(tt, x, file)

    def plot_data_from_amplitude_file(self, filename):
        fs = 200
        file = Path(filename)
        if re.match(r'^\.txt$', file.suffix):
            x = ingest.read_gui_amplitude_file_data(file)
            n = len(x)
            tt = np.linspace(0, n / fs, n)
        self.stim_pulse_window.set_recording(None)
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
        slice_q = dm.RecordingSlice.select().join(dm.RecordingFile)\
            .where(dm.RecordingFile.filename == file.name)
        if slice_q.count() == 1:
            slice = slice_q.get()
            self.start_slice_input.clear()
            self.length_slice_input.clear()
            self.start_slice_input.insert(str(slice.start))
            self.length_slice_input.insert(str(slice.length))
            highlight_start = slice.start
            highlight_stop = slice.start + slice.length
            if slice.recording_rejected:
                c = 'red'
            else:
                c = 'green'
            self.time_plot.axes.fill_betweenx([min(x), max(x)],
                                              highlight_start,
                                              highlight_stop,
                                              color=c, alpha=0.2)
        self.time_plot.fig.tight_layout()
        self.time_plot.draw()

        fs = 20000
        if slice_q.count() == 1:
            start_sample = int(highlight_start * fs)
            end_sample = int((highlight_stop) * fs)
            x = x[start_sample:end_sample]
        self.psd_plot.axes.cla()
        if len(x) > fs:
            f_signal, spectrum_signal = \
                signal.welch(x, fs, nperseg=self.welch_window_length*fs)
            self.psd_plot.axes.plot(f_signal, spectrum_signal)
            self.psd_plot.axes.set_xlim([0, 150])
            max_150 = max(spectrum_signal[f_signal <= 150]) * 1.05
            self.psd_plot.axes.set_ylim([0, max_150])
        self.psd_plot.axes.set_xlabel('Frequency [Hz]')
        self.psd_plot.axes.set_title('PSD')

        self.psd_plot.fig.tight_layout()
        self.psd_plot.draw()
        if self.stim_pulse_window.isVisible():
            self.stim_pulse_window.update_display()

    def display_time_points(self, x: list[float], color: str) -> plt.Line2D:
        y = [0 for e in x]
        line = self.time_plot.axes.plot(x, y, color=color, marker=6,
                                        linestyle=' ')
        self.time_plot.draw()
        return line[0]

    def hide_time_points(self, line: plt.Line2D) -> None:
        line.remove()
        self.time_plot.draw()

    def populate_file_list(self):
        dir = Path(self.file_dir)
        extension_regexp = re.compile(r'.*\.(mat|txt|bin)$')

        rat = self.rat_selected
        condition = self.condition_selected
        stim = self.stim_selected

        if rat is None and condition is None and stim is None:
            file_list = [file.name for file in dir.iterdir() if
                         re.match(extension_regexp, file.name)]
        else:
            matching_files = dm.files_matching_filter(rat, condition, stim)
            file_list = [file.name for file in dir.iterdir() if
                         re.match(extension_regexp, file.name) and
                         file.name in matching_files]
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

    def reject_file(self):
        filename = Path(self.file_list[self.current_file])
        file_key = filename.stem
        max_t = np.ceil(max(self.time_plot.axes.lines[0].get_xdata()))
        dm.update_slice(filename.name, 0, max_t, True)
        if file_key in self.time_slices:
            self.time_slices[file_key]['reject'] = True
            self.time_slices[file_key]['start'] = 0.0
            self.time_slices[file_key]['length'] = max_t
        else:
            slice = {
                'start': 0.0,
                'length': max_t,
                'reject': True
            }
            self.time_slices[file_key] = slice
        self.file_list_widget.selectedItems()[0].setIcon(
                QtGui.QIcon(r"stop.png"))
        with open(self.time_slices_file, mode='wb') as f:
            pickle.dump(self.time_slices, f)
        full_filename = Path(self.file_dir) / filename.name
        self.plot_data_from_file(full_filename)

    def update_selected_slice(self):
        filename = Path(self.file_list[self.current_file])
        start_raw = self.start_slice_input.text()
        length_raw = self.length_slice_input.text()

        try:
            start = float(start_raw)
        except ValueError:
            start = None
        try:
            length = float(length_raw)
        except ValueError:
            length = None
        max_t = np.ceil(max(self.time_plot.axes.lines[0].get_xdata()))

        if start is not None and start < max_t:
            if length is None or start + length > max_t:
                length = max_t - start
            self.time_slices[filename.stem] = {'start': start,
                                               'length': length}
            dm.update_slice(filename.name, start, length, False)
            self.file_list_widget.selectedItems()[0].setIcon(
                QtGui.QIcon(r"scissors.png"))
        else:
            if filename.stem in self.time_slices:
                self.time_slices.pop(filename.stem)
            self.file_list_widget.selectedItems()[0].setIcon(QtGui.QIcon(None))
            s = dm.RecordingSlice.select().join(dm.RecordingFile)\
                .where(dm.RecordingFile.filename == filename.name)
            if s.count() == 1:
                s.get().delete_instance()
        with open(self.time_slices_file, mode='wb') as f:
            pickle.dump(self.time_slices, f)
        full_filename = Path(self.file_dir) / filename.name
        self.plot_data_from_file(full_filename)

    def show_stim_pulse_window(self):
        self.stim_pulse_window.update_display()
        self.stim_pulse_window.show()

    def get_current_filename(self) -> str:
        if self.current_file is None:
            return ''
        return self.file_list[self.current_file].split('.')[0]

    def subtract_template(self, template: PulseTemplate) -> None:
        self.pulse_template = template
        self.subtract_pulse = True
        filename = Path(self.file_list[self.current_file])
        full_filename = Path(self.file_dir) / filename.name
        if self.stim_pulse_window.start_markers is not None:
            self.stim_pulse_window.toggle_start_on_main_plot()
        if self.stim_pulse_window.end_markers is not None:
            self.stim_pulse_window.toggle_end_on_main_plot()
        self.plot_data_from_file(full_filename)


class StimPulseWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(StimPulseWindow, self).__init__(parent)

        self.setWindowTitle('Stimulus pulse')
        self.recording = None
        self.template_length = 15
        self.template_align = 'max'
        self.start_offset = 0
        self.highpass_cutoff = 1
        self.start_markers: plt.Line2D = None
        self.end_markers = None
        self.template = np.zeros(self.template_length)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)

        highpass_filter_area = QtWidgets.QHBoxLayout()
        highpass_filter_area.addWidget(QtWidgets.QLabel('Highpass filter:'))
        self.highpass_filter_method = QtWidgets.QComboBox()
        self.highpass_filter_method.insertItems(0, ['None',
                                                    '1 Hz (interpolated)',
                                                    '10 Hz (interpolated)'])
        self.highpass_filter_method.currentIndexChanged.connect(
            self.update_highpass_filter)
        highpass_filter_area.addWidget(self.highpass_filter_method)

        template_align_area = QtWidgets.QHBoxLayout()
        self.align_method = QtWidgets.QComboBox()
        self.align_method.insertItems(0, ['Max', 'Start'])
        self.align_method.currentIndexChanged.connect(
            self.update_template_align)
        template_align_area.addWidget(QtWidgets.QLabel('Template alignment:'))
        template_align_area.addWidget(self.align_method)

        template_start_area = QtWidgets.QHBoxLayout()
        self.start_type = QtWidgets.QComboBox()
        self.start_type.insertItems(0, ['From data with offset'])
        self.start_input = QtWidgets.QLineEdit()
        self.start_input.setText(str(self.start_offset))
        self.start_input.editingFinished.connect(self.update_start_offset)
        self.toggle_start_button = QtWidgets.QPushButton(
            'Show on main plot')
        self.toggle_start_button.clicked.connect(
            self.toggle_start_on_main_plot)
        template_start_area.addWidget(QtWidgets.QLabel('Template start'),
                                      stretch=1)
        template_start_area.addWidget(self.start_type, stretch=1)
        template_start_area.addWidget(self.start_input, stretch=2)
        template_start_area.addWidget(self.toggle_start_button, stretch=1)

        template_end_area = QtWidgets.QHBoxLayout()
        self.end_type = QtWidgets.QComboBox()
        # self.end_type.insertItems(0, ['Fixed length', 'From data'])
        # self.end_type.currentIndexChanged.connect(self.update_template_length)
        self.end_type.insertItems(0, ['Fixed length'])
        self.end_input = QtWidgets.QLineEdit()
        self.end_input.setText(str(self.template_length))
        self.end_input.editingFinished.connect(self.update_template_length)
        self.toggle_end_button = QtWidgets.QPushButton(
            'Show on main plot')
        self.toggle_end_button.clicked.connect(
            self.toggle_end_on_main_plot)
        template_end_area.addWidget(QtWidgets.QLabel('Template end'),
                                    stretch=1)
        template_end_area.addWidget(self.end_type, stretch=1)
        template_end_area.addWidget(self.end_input, stretch=2)
        template_end_area.addWidget(self.toggle_end_button, stretch=1)

        channel_select_area = QtWidgets.QHBoxLayout()
        self.selected_channel = QtWidgets.QComboBox()
        self.selected_channel.insertItems(0, ['All', 'Mean'])
        self.selected_channel.currentIndexChanged.connect(self.update_display)
        channel_select_area.addWidget(QtWidgets.QLabel('Selected channel'),
                                      stretch=2)
        channel_select_area.addWidget(self.selected_channel, stretch=3)

        pulse_select_area = QtWidgets.QHBoxLayout()
        self.selected_pulse = QtWidgets.QComboBox()
        self.selected_pulse.insertItems(0, ['Mean', 'All'])
        self.selected_pulse.currentIndexChanged.connect(self.update_display)
        pulse_select_area.addWidget(QtWidgets.QLabel('Selected pulses'),
                                    stretch=2)
        pulse_select_area.addWidget(self.selected_pulse, stretch=3)

        subtract_template_area = QtWidgets.QHBoxLayout()
        self.subtract_template_button = QtWidgets.QPushButton()
        self.subtract_template_button.setText("Subtract template")
        self.subtract_template_button.clicked.connect(
            self.subtract_current_template)
        subtract_template_area.addWidget(self.subtract_template_button,
                                         stretch=5)

        plot_area = QtWidgets.QVBoxLayout()
        plot_area.addLayout(highpass_filter_area)
        plot_area.addLayout(template_align_area)
        plot_area.addLayout(template_start_area)
        plot_area.addLayout(template_end_area)
        plot_area.addLayout(channel_select_area)
        plot_area.addLayout(pulse_select_area)
        plot_area.addWidget(self.toolbar)
        plot_area.addWidget(self.canvas)
        plot_area.addLayout(subtract_template_area)

        widget = QtWidgets.QWidget()
        widget.setLayout(plot_area)

        self.setCentralWidget(widget)

    def error_box(self, text: str):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText('Error')
        msg.setInformativeText(text)
        msg.setWindowTitle('Error')
        msg.exec()

    def update_template_align(self):
        self.template_align = self.align_method.currentText().lower()
        self.update_display()

    def update_highpass_filter(self):
        values = {
            'None': None,
            '1 Hz (interpolated)': 1,
            '10 Hz (interpolated)': 10
            }
        selected = self.highpass_filter_method.currentText()
        self.highpass_cutoff = values[selected]
        self.update_display()

    def update_start_offset(self):
        try:
            new_offset = int(self.start_input.text())
            self.start_offset = new_offset
            self.update_display()
            if self.start_markers is not None:
                self.toggle_start_on_main_plot()
                self.toggle_start_on_main_plot()
            if self.end_markers is not None:
                self.toggle_end_on_main_plot()
                self.toggle_end_on_main_plot()
        except ValueError:
            self.start_input.setFocus()
            self.start_input.selectAll()
            self.error_box('Start offset must be integer')
        except AttributeError:
            self.start_input.setText(str(self.start_offset))
            self.error_box('Load a recording first')

    def update_template_length(self):
        try:
            new_length = int(self.end_input.text())
            self.template_length = new_length
            self.update_display()
            if self.start_markers is not None:
                self.toggle_start_on_main_plot()
                self.toggle_start_on_main_plot()
            if self.end_markers is not None:
                self.toggle_end_on_main_plot()
                self.toggle_end_on_main_plot()
        except ValueError:
            self.end_input.setFocus()
            self.end_input.selectAll()
            self.error_box('Template length must be integer')
        except AttributeError:
            self.end_input.setText(str(self.template_length))
            self.error_box('Load a recording first')

    def toggle_markers(self, points, color, line, button):
        if line is None:
            line = self.parent().display_time_points(points, color)
            button.setText('Hide on main plot')
        else:
            self.parent().hide_time_points(line)
            line = None
            button.setText('Show on main plot')
        return line

    def toggle_start_on_main_plot(self):
        if self.recording is None:
            raise AttributeError
        if self.recording.slice is not None:
            slice_start = self.recording.slice[0]
            slice_end = self.recording.slice[1]
        else:
            slice_start = 0
            slice_end = self.recording.electrode_data.shape[1]\
                * self.recording.dt
        start_offset_seconds = self.start_offset * self.recording.dt
        points = [e[0] + start_offset_seconds
                  for e in self.recording.pulse_periods
                  if slice_start <= e[0] + start_offset_seconds <= slice_end]
        self.start_markers = self.toggle_markers(points, 'green',
                                                 self.start_markers,
                                                 self.toggle_start_button)

    def toggle_end_on_main_plot(self):
        if self.recording is None:
            raise AttributeError
        if self.recording.slice is not None:
            slice_start = self.recording.slice[0]
            slice_end = self.recording.slice[1]
        else:
            slice_start = 0
            slice_end = self.recording.electrode_data.shape[1]\
                * self.recording.dt
        end_offset = (self.start_offset + self.template_length)\
            * self.recording.dt
        points = [e[0] + end_offset
                  for e in self.recording.pulse_periods
                  if slice_start <= e[0] + end_offset <= slice_end]
        self.end_markers = self.toggle_markers(points, 'red',
                                               self.end_markers,
                                               self.toggle_end_button)

    def update_display(self):
        self.canvas.axes.cla()

        plot_title = self.parent().get_current_filename()
        if self.recording is None:
            self.plot_message_center('No Recording')
        elif self.recording.filename != plot_title:
            self.plot_message_center('Filename does not match the Recording')
        else:
            if len(self.recording.pulse_periods) == 0:
                self.plot_message_center('No stim in this file')
            else:
                if self.selected_pulse.currentText() == 'Mean':
                    channels = self.selected_channel.currentText().lower()
                    slice = self.recording.slice
                    t_length = self.template_length
                    align = self.template_align
                    hi_cutoff = self.highpass_cutoff
                    template = process.create_pulse_template(self.recording,
                                                             t_length,
                                                             self.start_offset,
                                                             align,
                                                             slice,
                                                             channels,
                                                             hi_cutoff)
                    if len(template.shape) == 1:
                        self.canvas.axes.plot(template)
                    else:
                        for i in range(template.shape[0]):
                            self.canvas.axes.plot(template[i, :])
                        self.canvas.axes.legend(range(1, template.shape[0]+1))
                    self.template = template
                elif self.selected_pulse.currentText() == 'All':
                    d = np.mean(self.recording.electrode_data, axis=0)
                    fs = int(1/self.recording.dt)
                    for i, p in enumerate(self.recording.pulse_periods):
                        if i % 10 == 0:
                            if (p[0] < self.recording.slice[0] or
                                    p[1] > self.recording.slice[1]):
                                continue
                            s = int(p[0] * fs) + self.start_offset
                            e = s + self.template_length
                            if (self.template_align == 'max' and
                                    len(d[s:e]) > 0):
                                max_location = np.argmax(d[s:e])
                                s = s + max_location\
                                    - int(np.floor(self.template_length / 2))
                                e = s + self.template_length
                            if (s > 0 and e < len(d) and s < len(d)):
                                self.canvas.axes.plot(d[s:e], color='grey',
                                                      linewidth=0.6)
        self.canvas.axes.set_title(plot_title)
        self.canvas.draw()
        self.canvas.flush_events()

    def plot_message_center(self, text: str) -> None:
        self.canvas.axes.set_xlim([0, 1])
        self.canvas.axes.set_ylim([0, 1])
        self.canvas.axes.text(0.5, 0.5, text, ha='center', va='center')

    def set_recording(self, data: ingest.Recording) -> None:
        self.recording = data
        if data is not None:
            self.recording.filename = str(pathlib.Path(data.filename).stem)

    def subtract_current_template(self):
        if self.recording is None:
            self.error_box("No recording selected")
            return
        elif len(self.recording.pulse_periods) == 0:
            self.error_box("No stimulation in this recording")
            return
        start = [int(e[0] / self.recording.dt) + self.start_offset
                 for e in self.recording.pulse_periods]
        if self.selected_channel.currentText().lower() == 'mean':
            channels = 1
        elif self.selected_channel.currentText().lower() == 'all':
            channels = self.recording.electrode_data.shape[0]
        template = PulseTemplate(self.template_length, channels,
                                 self.template, start, self.template_align)
        self.parent().subtract_template(template)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
