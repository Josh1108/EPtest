from functools import wraps

CLASSIFIER_REGISTRY = {}
TOKEN_POOLER_REGISTRY = {}
SPAN_POOLER_REGISTRY = {}
REPORTER_REGISTRY = {}
DATA_LOADER_REGISTRY = {}
PROBE_REGISTRY = {}
ENCODER_REGISTRY = {}
DATA_FILTER_REGISTRY = {}
CLASSIFIER = "classifier"
TOKEN_POOLER = "token_pooler"
SPAN_POOLER = "span_pooler"
REPORTER = "reporter"
DATA_LOADER = "data_loader"
PROBE = "probe"
ENCODER = "encoder"
DATA_FILTER = "data_filter"


def optional_params(func):
    """Allow a decorator to be called without parentheses if no kwargs are given.

    parameterize is a decorator, function is also a decorator.
    """

    @wraps(func)
    def wrapped(*args, **kwargs):
        """If a decorator is called with only the wrapping function just execute the real decorator.
           Otherwise return a lambda that has the args and kwargs partially applied and read to take a function as an
           argument.

        *args, **kwargs are the arguments that the decorator we are parameterizing is called with.

        the first argument of *args is the actual function that will be wrapped
        """
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return func(args[0])
        return lambda x: func(x, *args, **kwargs)

    return wrapped


@optional_params
def register(cls, _type: str, _name: str):
    if _type == CLASSIFIER:
        CLASSIFIER_REGISTRY[_name] = cls
    elif _type == TOKEN_POOLER:
        TOKEN_POOLER_REGISTRY[_name] = cls
    elif _type == SPAN_POOLER:
        SPAN_POOLER_REGISTRY[_name] = cls
    elif _type == REPORTER:
        REPORTER_REGISTRY[_name] = cls
    elif _type == DATA_LOADER:
        DATA_LOADER_REGISTRY[_name] = cls
    elif _type == PROBE:
        PROBE_REGISTRY[_name] = cls
    elif _type == ENCODER:
        ENCODER_REGISTRY[_name] = cls
    elif _type == DATA_FILTER:
        DATA_FILTER_REGISTRY[_name] = cls
    else:
        raise RuntimeError(f"No suitable registry found for type {_type}")
    return cls
