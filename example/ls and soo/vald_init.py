import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

pic_type = ".pdf"
dir_save = '../figures/'

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

config = {
    'text.usetex': True,
    # "font.size": 13,
    "mathtext.fontset": 'stix',
}
a = 4.5
b = 3.5

rcParams.update(config)
