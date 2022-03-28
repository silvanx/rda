from datetime import datetime
from pathlib import Path
import pathlib
import re
import ratdata.ingest as ingest
import dateutil.parser as dparser
from peewee import SqliteDatabase, AutoField, DateField, TextField, \
                   ForeignKeyField, Model, DatabaseProxy, IntegerField, \
                   FloatField, SelectQuery
import numpy as np


database_proxy = DatabaseProxy()


class Rat(Model):
    rat_id = AutoField()
    date_start = DateField()
    date_end = DateField()
    label = TextField()
    group = TextField()

    class Meta:
        database = database_proxy


class RecordingFile(Model):
    file_id = AutoField()
    dirname = TextField()
    filename = TextField()
    recording_date = DateField()
    rat = ForeignKeyField(Rat, field='rat_id')
    condition = TextField()

    class Meta:
        database = database_proxy


class StimSettings(Model):
    stim_settings_id = AutoField()
    recording_file = ForeignKeyField(RecordingFile, field='file_id',
                                     backref='stim')
    max_amplitude = IntegerField()
    stim_type = TextField()

    class Meta:
        database = database_proxy


class RecordingBaseline(Model):
    recording = ForeignKeyField(RecordingFile, backref='baseline')
    baseline = ForeignKeyField(RecordingFile)

    class Meta:
        database = database_proxy


class RecordingPower(Model):
    recording = ForeignKeyField(RecordingFile, backref='power', unique=True)
    beta_power = FloatField()
    total_power = FloatField()

    class Meta:
        database = database_proxy


def db_connect(db_filename: str) -> None:
    database_proxy.initialize(SqliteDatabase(db_filename))


def db_create_tables() -> None:
    database_proxy.connect()
    database_proxy.create_tables([Rat, RecordingFile, StimSettings,
                                  RecordingBaseline, RecordingPower])
    database_proxy.close()


def find_all_recording_files_dir(dirname: str) -> list[str]:
    dir = Path(dirname)
    f_re = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}'
                      r'.*(\.(mat|bin)|-amplitude.txt)$')
    filelist = [file.name for file in dir.iterdir() if f_re.match(file.name)]
    return filelist


def find_processed_files_dir(dirname: str) -> list[str]:
    dirname_full = str(pathlib.Path(dirname).absolute())
    query = RecordingFile.select().where(RecordingFile.dirname == dirname_full)
    result = query.execute()
    return [f.filename for f in result]


def find_unprocessed_files_dir(dirname: str) -> list[str]:
    all_files = find_all_recording_files_dir(dirname)
    processed_files = find_processed_files_dir(dirname)
    return list(set(all_files) - set(processed_files))


def process_single_file(filename: str) -> None:
    add_file_to_database(filename)


def db_get_rat(timestamp: str, rat_label: str) -> Rat:
    recording_date = dparser.parse(timestamp).date()
    query = (Rat.select().where((Rat.date_start <= recording_date)
                                & (recording_date <= Rat.date_end)
                                & (Rat.label == rat_label)))
    result = query.get()
    return result


def db_get_recording(filename: str) -> RecordingFile:
    query = RecordingFile.select().where(RecordingFile.filename == filename)
    result = query.get()
    return result


def add_file_to_database(filename: str) -> None:
    f = pathlib.Path(filename)
    timestamp, rat, condition = ingest.extract_info_from_filename(filename)

    if None in (timestamp, rat, condition):
        print("Error processing {}, skipping".format(filename))
        return

    RecordingFile.create(
        dirname=f.parent.absolute(),
        filename=f.name,
        recording_date=dparser.parse(timestamp),
        rat=db_get_rat(timestamp, rat),
        condition=condition
    )


def add_rat_to_database(date_start: datetime.date, date_end: datetime.date,
                        label: str, group: str) -> tuple[Rat, bool]:
    result, created = Rat.get_or_create(date_start=date_start,
                                        date_end=date_end, label=label,
                                        group=group)
    return result, created


def add_stim_params_to_database(filename: str, amplitude: int,
                                stim_type: str) -> tuple[StimSettings, bool]:
    recording_file = db_get_recording(filename)
    result, created = StimSettings.get_or_create(recording_file=recording_file,
                                                 max_amplitude=amplitude,
                                                 stim_type=stim_type)
    return result, created


def process_new_files(dirname: str) -> None:
    filelist = find_unprocessed_files_dir(dirname)
    num_files = len(filelist)
    for index, file in enumerate(filelist):
        f = "{}/{}".format(dirname, file)
        print('[{} / {}] {}'.format(index + 1, num_files, file))
        process_single_file(f)


def select_recordings_for_rat(rat: Rat, condition: str,
                              stim_type: str) -> SelectQuery:
    query = RecordingFile.select() \
        .join(StimSettings).where((RecordingFile.rat == rat) &
                                  (RecordingFile.condition == condition) &
                                  (StimSettings.stim_type == stim_type))
    return query


def get_all_recording_dates() -> list[datetime.date]:
    query = RecordingFile.select().group_by(RecordingFile.recording_date)
    dates = [r.recording_date for r in query]
    return dates


def get_all_recording_dates_nostim() -> list[datetime.date]:
    query = RecordingFile.select()\
        .join(StimSettings)\
        .where(StimSettings.stim_type == 'nostim')\
        .group_by(RecordingFile.recording_date)
    dates = [r.recording_date for r in query]
    return dates


def get_electrode_data_from_recording(rec: RecordingFile) -> tuple[np.ndarray,
                                                                   float]:
    file_fullpath = str(pathlib.Path(rec.dirname) / rec.filename)
    recording_data = ingest.read_mce_matlab_file(file_fullpath)
    return (recording_data.electrode_data, recording_data.dt)


def get_rat_labels() -> list[str]:
    query = Rat.select().order_by(Rat.label)
    rat_list = [r.label for r in query]
    rat_list.insert(0, None)
    return rat_list
