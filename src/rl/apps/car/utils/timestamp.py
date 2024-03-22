from datetime import datetime


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
