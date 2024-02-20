from datetime import datetime


def timestamp():
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
