
logging_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "custom": {
            # More format options are available in the official
            # `documentation <https://docs.python.org/3/howto/logging-cookbook.html>`_
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    # Any INFO level msg will be printed to the console
    "handlers": {
        "console": {
            "level": "DEBUG",
            "formatter": "custom",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },

     "file_handler": {
            "level": "DEBUG",
            "formatter": "custom",
            "class": "logging.FileHandler",
            "filename": "autosklearn.log",
        },

    },

    "loggers": {
        "": {  # root logger
            "level": "DEBUG",
            "handlers": ["console", 'file_handler'],
        },

        # "Client-EnsembleBuilder": {
        #     "level": "DEBUG",
        #     "handlers": ["console", 'file_handler'],
        # },

    },
}
