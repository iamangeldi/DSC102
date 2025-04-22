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

    # usecols(): We use this in order to load the needed columns, although python does initially read the whole row then drops
    # useless columns this speeds up the proccess in future computations due to its significantly less memory allocation
    # dtype: Tells Dask how to prse columns initially, we can also avoid expensive data types if its possible.
    # blocksize: Lets us parse each "task" in block of 64MB which makes parrallelism of work more efficient every core can optimize
    # workload, no idle time, we dont load everything in one go.
    df = dd.read_csv(path_to_user_reviews_csv, usecols=['reviewerID','overall','unixReviewTime','helpful'],
        dtype={'reviewerID':'object','overall':'float64','helpful':'object'},
        blocksize='64MB')

    # Strip is useful because we can parse the data once without having to call for helpful column twice in helpful and total votes
    helpful = df['helpful'].str.strip('[]').str.split(',')
    df['helpful_votes'] = helpful.str.get(0).astype('int64')
    df['total_votes'] = helpful.str.get(1).astype('int64')
    df = df.drop('helpful', axis=1)

    # Same review tranformation, can't really get better than that, if you want to tarnsform inside the call of the df seems like
    # it will be much slower
    df['reviewing_year'] = dd.to_datetime(df['unixReviewTime'], unit='s').dt.year
    df = df.drop('unixReviewTime', axis=1)

    # THIS IS A GAME CHANGER, persist() basically saves the work into worker memory so that it easier for aggregatiosn to take place, 
    # to put it in other words, it takes the "ready dough" in the form of the previous column transformations, and puts in the hands of 
    # bakers that then add their own twist and bakes the cookies
    df.persist()

    # Overall column can be used as both a count for unique products rated, and average rating. Intrestingly, changing the order of 
    # mean and count has substantial effect on performance, as mean + count yields 261 secs and count + mean yields 281 secs more 
    # research is needed, on additional review the timing is always around 280 ish secs, wonder why that is happening?
    users = df.groupby('reviewerID').agg({
        'overall':       ['mean', 'count'],
        'reviewing_year':['min'],
        'helpful_votes': ['sum'],
        'total_votes':   ['sum']
    })
    users.columns = ['avg_ratings', 'number_products_rated',
      'reviewing_since','helpful_votes','total_votes'
    ]
    
    submit = users.describe().compute().round(2)    
    with open('results_PA0.json', 'w') as outfile: 
        json.dump(json.loads(submit.to_json()), outfile)