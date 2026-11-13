class ThreadStopEventException(Exception):
    """Исключение возникает, когда stop_event потока становится set()."""
    pass
