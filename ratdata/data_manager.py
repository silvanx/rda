from datetime import datetime
from pathlib import Path
import pathlib
import re
import ratdata.ingest as ingest
import dateutil.parser as dparser
from peewee import SqliteDatabase, AutoField, DateField, TextField, \
                   ForeignKeyField, Model, DatabaseProxy, IntegerField, \
                   FloatField, SelectQuery, BooleanField
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
    oof_exponent = FloatField()
    oof_constant = FloatField()

    class Meta:
        database = database_proxy


class RecordingSlice(Model):
    recording = ForeignKeyField(RecordingFile, backref='slice', unique=True)
    start = FloatField()
    length = FloatField()
    recording_rejected = BooleanField(default=False)
    updated = BooleanField(default=False)

    class Meta:
        database = database_proxy


def db_connect(db_filename: str) -> None:
    database_proxy.initialize(SqliteDatabase(db_filename))


def db_create_tables() -> None:
    database_proxy.connect()
    database_proxy.create_tables([Rat, RecordingFile, StimSettings,
                                  RecordingBaseline, RecordingPower,
                                  RecordingSlice])
    database_proxy.close()


def update_slice(filename: str, start: float, length: float,
                 reject: bool = False) -> None:
    try:
        rec = RecordingFile.get(filename=filename)
    except RecordingFile.DoesNotExist:
        print("ERROR: file %s is not in the database" % filename)
        return
    if rec.slice.count() == 1:
        RecordingSlice.update(start=start, length=length,
                              recording_rejected=reject, updated=True)\
                      .where(RecordingSlice.recording == rec).execute()
    elif rec.slice.count() == 0:
        RecordingSlice.insert(recording=rec, start=start, length=length,
                              recording_rejected=reject,
                              updated=True).execute()
    else:
        print("ERROR: multiple slices assigned to recording %s: %d" %
              (filename, rec.slice.count()))


def is_recording_rejected(filename: str) -> bool:
    q = RecordingFile.select()\
        .where(RecordingFile.filename == filename)
    if (q.count() == 1 and
            q.get().slice.count() == 1 and
            q.get().slice.get().recording_rejected):
        return True
    else:
        return False


def is_recording_sliced(filename: str) -> bool:
    q = RecordingSlice.select().join(RecordingFile)\
        .where(RecordingFile.filename == filename)
    return q.count() == 1


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


def get_electrode_data_from_recording(rec: RecordingFile,
                                      select_slice: bool) -> tuple[np.ndarray,
                                                                   float]:
    file_fullpath = str(pathlib.Path(rec.dirname) / rec.filename)
    recording_data = ingest.read_mce_matlab_file(file_fullpath)
    data = recording_data.electrode_data
    dt = recording_data.dt
    if select_slice and rec.slice.count() == 1:
        slice = rec.slice.get()
        start = int(slice.start / dt)
        end = int((slice.start + slice.length) / dt)
        data = data[:, start:end]
        return (data, dt, (slice.start, slice.start + slice.length))
    else:
        return (data, dt, (0, data.shape[1] * dt))


def get_rat_labels() -> list[str]:
    query = Rat.select().order_by(Rat.label)
    result = [r.label for r in query]
    result.insert(0, None)
    return result


def get_stim_types() -> list[str]:
    query = StimSettings.select().group_by(StimSettings.stim_type)
    result = [e.stim_type for e in query]
    result.insert(0, None)
    return result


def get_condition_labels() -> list[str]:
    query = RecordingFile.select().group_by(RecordingFile.condition)
    result = [e.condition for e in query]
    result.insert(0, None)
    return result


def get_recording_slice(filename: str) -> tuple[float, float]:
    query = RecordingSlice.select().join(RecordingFile)\
        .where(RecordingFile.filename == filename)
    if query.count() == 1:
        slice = query.get()
        return (slice.start, slice.start + slice.length)
    else:
        return None


def upsert_power_record(f: ingest.Recording, recording_beta_power: float,
                        recording_total_power: float,
                        oof_m: float, oof_b: float) -> int:
    q = RecordingPower.select().join(RecordingFile)\
        .where(RecordingFile.filename == f.filename)
    if q.count() == 0:
        rec = RecordingFile.get(filename=f.filename)
        rp = RecordingPower.insert(recording=rec,
                                   beta_power=recording_beta_power,
                                   total_power=recording_total_power,
                                   oof_exponent=oof_m,
                                   oof_constant=np.e**oof_b)
        rp_id = rp.execute()
        RecordingSlice.update(updated=False)\
            .where(RecordingSlice.recording == rec).execute()
    elif q.count() == 1:
        rec = q.get().recording
        rp_id = RecordingPower.update(beta_power=recording_beta_power,
                                      total_power=recording_total_power,
                                      oof_exponent=oof_m,
                                      oof_constant=np.e**oof_b)\
                              .where(RecordingPower.recording == rec)\
                              .execute()
        RecordingSlice.update(updated=False)\
            .where(RecordingSlice.recording == rec).execute()
    else:
        msg = 'ERROR: found %d records for file %s' %\
            (q.count(), f.filename)
        print(msg)
        rp_id = 0
    return rp_id
