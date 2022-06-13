import pandas as pd
import pickle
from datetime import datetime, timedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from random import randint

from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner

def read_data(path):
    df = pd.read_parquet(path)
    return df

def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    df_loc = "df_" + str(randint(1, 1000)) + ".pkl"

    dump_pickle(df, df_loc)

    return df_loc

@task
def train_model(df_loc, categorical, date):
    df = load_pickle(df_loc)
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")

    lr_loc = f"model-{date}.pkl"
    dv_loc = f"dv-{date}.pkl"
    dump_pickle(lr, lr_loc)
    dump_pickle(dv, dv_loc)

    return lr_loc, dv_loc

@task
def run_model(df_loc, categorical, dv_loc, lr_loc):
    df = load_pickle(df_loc)
    dv = load_pickle(dv_loc)
    lr = load_pickle(lr_loc)
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date=None):
    if date is None:
        date = datetime.today()
    else:
        date = datetime.strptime(date, '%Y-%m-%d')
    
    year, month, day = date.timetuple()[:3]

    train_date = datetime(year, month-2, day)
    val_date = datetime(year, month-1, day)
    
    return f"../data/fhv_tripdata_{train_date.strftime('%Y-%m')}.parquet", f"../data/fhv_tripdata_{val_date.strftime('%Y-%m')}.parquet"


@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):

    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']
    
    global logger
    logger = get_run_logger()

    df_train = read_data(train_path)
    df_train_processed_loc = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed_loc = prepare_features(df_val, categorical, False)

    # train the model
    lr_loc, dv_loc = train_model(df_train_processed_loc, categorical, date).result()
    run_model(df_val_processed_loc, categorical, dv_loc, lr_loc)

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)