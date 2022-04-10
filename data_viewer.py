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


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.file_dir = 'data/mce_recordings'
        self.db_file = 'rat_data.db'

        dm.db_connect('rat_data.db')

        self.current_file = None
        self.rat_selected = None
        self.condition_selected = None
        self.stim_selected = None
        self.file_list = self.populate_file_list()

        self.all_conditions = dm.get_condition_labels()
        self.all_rats = dm.get_rat_labels()
        self.all_stims = dm.get_stim_types()
        # self.all_rats = [None, 'rat1', 'rat2', 'rat3', 'rat5']
        # self.all_conditions = [None, 'baseline', 'ST', 'CT', 'OFT']
        # self.all_stims = [None, 'nostim', 'continuous', 'on-off',
        #                   'random', 'proportional']

        self.time_slices_file = Path(self.file_dir) / 'time_slices.pickle'
        self.time_slices = ingest.read_file_slices(self.time_slices_file)

        self.time_plot = MplCanvas(self, width=5, height=4, dpi=100)
        self.psd_plot = MplCanvas(self, width=5, height=3, dpi=100)

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

        # Plotting areas
        plot_area = QtWidgets.QVBoxLayout()
        plot_area.addLayout(time_plot_controls)
        plot_area.addWidget(self.time_plot)
        plot_area.addWidget(toolbar_psd)
        plot_area.addWidget(self.psd_plot)
        plot_area.addLayout(nav_buttons)

        # File list on the right
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.setIconSize(QtCore.QSize(12, 12))
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

        self.setWindowTitle('Rat data analysis')
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
            if dm.is_recording_rejected(file):
                currentItem.setIcon(QtGui.QIcon(r"stop.png"))
            elif dm.is_recording_sliced(file):
                currentItem.setIcon(QtGui.QIcon(r"scissors.png"))
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
            self.stim_pulse_window.set_recording(data)
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
            f_signal, spectrum_signal = signal.welch(x, fs, nperseg=fs)
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


class StimPulseWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(StimPulseWindow, self).__init__(parent)

        self.setWindowTitle('Stimulus pulse')
        self.recording = None
        self.template_length = 15
        self.start_offset = 0
        self.start_markers: plt.Line2D = None
        self.end_markers = None

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # self.toolbar = NavigationToolbar(self.canvas, self)

        template_start_area = QtWidgets.QHBoxLayout()
        self.start_type = QtWidgets.QComboBox()
        self.start_type.insertItems(0, ['From data'])
        self.start_input = QtWidgets.QLineEdit()
        self.start_input.setText(str(self.start_offset))
        self.start_input.editingFinished.connect(self.update_start_offset)
        self.toggle_start_button = QtWidgets.QPushButton(
            'Show on main plot')
        self.toggle_start_button.clicked.connect(
            self.toggle_start_on_main_plot)
        template_start_area.addWidget(QtWidgets.QLabel('Template start'))
        template_start_area.addWidget(self.start_type)
        template_start_area.addWidget(self.start_input)
        template_start_area.addWidget(self.toggle_start_button)

        template_end_area = QtWidgets.QHBoxLayout()
        self.end_type = QtWidgets.QComboBox()
        self.end_type.insertItems(0, ['Fixed length', 'From data'])
        self.end_input = QtWidgets.QLineEdit()
        self.end_input.setText(str(self.template_length))
        self.end_input.editingFinished.connect(self.update_template_length)
        self.toggle_end_button = QtWidgets.QPushButton(
            'Show on main plot')
        self.toggle_end_button.clicked.connect(
            self.toggle_end_on_main_plot)
        template_end_area.addWidget(QtWidgets.QLabel('Template end'))
        template_end_area.addWidget(self.end_type)
        template_end_area.addWidget(self.end_input)
        template_end_area.addWidget(self.toggle_end_button)

        channel_select_area = QtWidgets.QHBoxLayout()
        self.selected_channel = QtWidgets.QComboBox()
        self.selected_channel.insertItems(0, ['All', 'Mean'])
        self.selected_channel.currentIndexChanged.connect(self.update_display)
        channel_select_area.addWidget(QtWidgets.QLabel('Selected channel'))
        channel_select_area.addWidget(self.selected_channel)

        plot_area = QtWidgets.QVBoxLayout()
        plot_area.addLayout(template_start_area)
        plot_area.addLayout(template_end_area)
        plot_area.addLayout(channel_select_area)
        # plot_area.addWidget(self.toolbar)
        plot_area.addWidget(self.canvas)

        widget = QtWidgets.QWidget()
        widget.setLayout(plot_area)

        self.setCentralWidget(widget)

    def error_box(text: str):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText('Error')
        msg.setInformativeText(text)
        msg.setWindowTitle('Error')
        msg.exec()

    def update_start_offset(self):
        try:
            new_offset = int(self.start_input.text())
            self.start_offset = new_offset
            offset_length_seconds = self.start_offset * self.recording.dt
            self.update_display()
            if self.start_markers is not None:
                self.parent().hide_time_points(self.start_markers)
                points = [e[0] + offset_length_seconds
                          for e in self.recording.pulse_periods]
                self.start_markers = self.parent().display_time_points(
                    points, 'green')
        except ValueError:
            self.error_box('Start offset must be integer')

    def update_template_length(self):
        try:
            new_length = int(self.end_input.text())
            self.template_length = new_length
            end_offset = (self.start_offset + self.template_length)\
                * self.recording.dt
            self.update_display()
            if self.end_markers is not None:
                self.parent().hide_time_points(self.end_markers)
                points = [e[0] + end_offset
                          for e in self.recording.pulse_periods]
                self.end_markers = self.parent().display_time_points(
                    points, 'red')
        except ValueError:
            self.error_box('Template length must be integer')

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
            return
        start_offset_seconds = self.start_offset * self.recording.dt
        points = [e[0] + start_offset_seconds
                  for e in self.recording.pulse_periods]
        self.start_markers = self.toggle_markers(points, 'green',
                                                 self.start_markers,
                                                 self.toggle_start_button)

    def toggle_end_on_main_plot(self):
        if self.recording is None:
            return
        end_offset = (self.start_offset + self.template_length)\
            * self.recording.dt
        points = [e[0] + end_offset
                  for e in self.recording.pulse_periods]
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
                channels = self.selected_channel.currentText().lower()
                template = process.create_pulse_template(self.recording,
                                                         self.template_length,
                                                         self.start_offset,
                                                         channels=channels)
                if len(template.shape) == 1:
                    self.canvas.axes.plot(template)
                else:
                    for i in range(template.shape[0]):
                        self.canvas.axes.plot(template[i, :])
                    self.canvas.axes.legend(range(1, template.shape[0] + 1))

        self.canvas.axes.set_title(plot_title)
        self.canvas.draw()
        self.canvas.flush_events()

    def plot_message_center(self, text: str) -> None:
        self.canvas.axes.set_xlim([0, 1])
        self.canvas.axes.set_ylim([0, 1])
        self.canvas.axes.text(0.5, 0.5, text, ha='center', va='center')

    def set_recording(self, data: ingest.Recording) -> None:
        self.recording = data
        data.filename = str(pathlib.Path(data.filename).stem)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
