import json
import ctypes
from dask.distributed import Client
import dask.dataframe as dd

def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def PA0(path_to_user_reviews_csv):
    client = Client()
    # Helps fix any memory leaks.
    client.run(trim_memory)
    client = client.restart()

    df = dd.read_csv(path_to_user_reviews_csv)

    df['helpful'] = df['helpful'].map(lambda x: eval(x), meta=('helpful', 'object'))
    df['helpful_votes'] = df['helpful'].map(lambda x: int(x[0]), meta=('helpful_votes', 'int'))
    df['total_votes'] = df['helpful'].map(lambda x: int(x[1]), meta=('total_votes', 'int'))

    df['reviewing_year'] = dd.to_datetime(df['unixReviewTime'], unit='s').dt.year

    users = df.groupby('reviewerID').agg({
        'asin': 'count',
        'overall': 'mean',
        'reviewing_year': 'min',
        'helpful_votes': 'sum',
        'total_votes': 'sum'
    }).reset_index()

    users.columns = [
        'reviewerID',
        'number_products_rated',
        'avg_ratings',
        'reviewing_since',
        'helpful_votes',
        'total_votes'
    ]
    
    
    submit = users.describe().compute().round(2)    
    with open('results_PA0.json', 'w') as outfile: 
        json.dump(json.loads(submit.to_json()), outfile)