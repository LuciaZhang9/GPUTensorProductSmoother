import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_data, write_convergence_table

files_2d_d = []

for i in range(1, 9):
    files_2d_d.append(f'/scratch/cucui/GPUTensorProductSmoothers/build_stokes/Convergence/Stokes_2D_RT{i}_Basic_Basic_GLOBAL_SchurTensorProduct_none_double_1s.log')

data_2d_d = read_data(files_2d_d)

write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_stokes/Convergence/table_2d", data_2d_d, 1, 0, "frac", 1)
