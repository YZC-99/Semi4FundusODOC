import os
import csv
from utils.general import get_config_from_file
from utils.log_metrics_write import logs2csv

path = 'experiments/Drishti-GS/cropped_sup256x256'
logs2csv(path)