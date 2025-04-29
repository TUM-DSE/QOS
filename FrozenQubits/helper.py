import pickle


def to_iterable(obj):
    ''' if obj is not an iterable (e.g., list, tuple, etc.), converts it to a tuple of size one'''
    try:
        iter(obj)
        return list(obj)
    except TypeError:
        return list([obj])



## saving/loading objects using Pickle 
def save_pickle(obj, filename):
    with open(filename, 'wb') as pkl:
        pickle.dump(obj, pkl)

def load_pickle(filename):
    with open(filename, 'rb') as pkl:
        return pickle.load(pkl)


