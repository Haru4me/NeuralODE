LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
            },
        },
        "handlers": {
            "file_handler": {
                "level": "INFO",
                "formatter": "default",
                "class": "logging.FileHandler",
                "mode": "a"
                },
            },
        "loggers": {
            "": {
                "handlers": ["file_handler"], "level": "INFO", "propagate": False}
            }
    }


CLIMATE_OPT = {
    'data/climate/air.mon.1981-2010.ltm.nc': 'air',
    'data/climate/soilw.mon.ltm.v2.nc': 'soilw',
    'data/climate/precip.mon.ltm.0.5x0.5.nc': 'precip'
}

SYN_COLS = ['ind', 'datetime', 't2m', 'td2m', 'ff', 'R12', 'phi']
AGRO_COLS = ['ind', 'dec', 'datetime', 'val_1', 'val_2']
