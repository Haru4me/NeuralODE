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

CAT_OPT = {
    'soil': {
        'tiff': 'data/agro/soil/so2015v2.tif',
        'description': 'data/agro/soil/2015_suborders_and_gridcode.txt'
    },
    'cover': {
        'tiff': 'data/agro/cover/GLOBCOVER_L4_200901_200912_V2.3.tif',
        'description': 'data/agro/cover/Globcover2009_Legend.xls'
    }
}

FEATURES_COLS = ['ind', 'dec', 'dec_next', 'ts', 'phi', 't2m', 'td2m', 'ff', 'R12',
                 'kult', 'soiltype', 'covertype', 'air', 'soilw', 'precip']

START_VAL_COLS = ['val_1', 'val_2']
TARGET_COLS = ['val_1_next', 'val_2_next']
