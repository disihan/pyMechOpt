import vald_init
from vald_init import pic_type, dir_save
import matplotlib.pyplot as plt
from pyMechOpt.multi_obj import mo_problem

from vald_mo import vald_mo

name = "nsga3"
name_cap = "NSGA3"

vald_mo(name, name_cap)
