import pickle
import pandas as pd
import numpy as np
import argparse

def output(df_result, output_file):
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def run(year, month):
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PUlocationID', 'DOlocationID']

    def read_data(filename):
        df = pd.read_parquet(filename)
        
        df['duration'] = df.dropOff_datetime - df.pickup_datetime
        df['duration'] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
        
        return df


    df = read_data(f'data/fhv_tripdata_{year:04d}-{month:02d}.parquet')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    mean = np.mean(y_pred)


    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    output_file = f'output-w4/{year:04d}-{month:02d}.parquet'

    #output(df_result, output_file)

    return mean

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year", type=int
    )
    parser.add_argument(
        "--month", type=int
    )
    args = parser.parse_args()

    mean = run(args.year, args.month)
    print(f"Mean is {mean}")