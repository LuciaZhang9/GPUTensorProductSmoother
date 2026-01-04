import numpy as np

def read_data(files):

    data = []

    for filename in files:
        # open the file and read its content
        with open(filename) as f:
            content = f.readlines()

        # extract the relevant data from the content
        local_data = []

        for line in content:
            line = line.strip()
            words = line.split()
            if line and words[0].isdigit() and words[1].isdigit():
                values = []
                for x in line.split():
                    try:
                        value = float(x)
                    except ValueError:
                        value = 0
                    values.append(value)
                local_data.append(values)
        data.append(np.array(local_data))
    # convert the data to a numpy array
    data = np.array(data, dtype=object)

    return data

def write_convergence_table(filename, dataset, deg_start, sm, it_type, n_kernels):
    
    n_p = dataset.shape[0]
    n_l = int(dataset[0].shape[0] / n_kernels)

    if it_type == "it":
        col = 7
    elif it_type == "frac":
        col = 8

    file1 = filename + ".txt"
    
    # Open the file in write mode
    with open(file1, "w") as f:

        # Write the LaTeX table header
        f.write("\\begin{table}[tp]\n")
        f.write("\\centering\n")
        f.write("\\renewcommand{\\arraystretch}{1.5}\n")
        f.write("\\begin{tabular}{c|" + "c"*n_p + "}\n")
        f.write("\\hline\n")

        f.write("$L$ &" + " & ".join(["$\mathbb{Q}_{" + str(x) + "}$" for x in range(deg_start,n_p+deg_start)]) + " \\\\\n")
        f.write("\\hline\n")

        # Write the table data
        for i in range(n_l):
            row_val = []
            row_val.append(int(dataset[0][i,0]))
            for k in range(n_p):
                shift = int(dataset[k].shape[0] / n_kernels * sm)
                if i < dataset[k].shape[0] / n_kernels:
                    val = int(dataset[k][i + shift,col]) if type == "it" else round(dataset[k][i + shift,col],1)
                else :
                    val = "---"
                row_val.append(val)
            f.write(" & ".join([str(x) for x in row_val]) + " \\\\\n")

        # Write the LaTeX table footer
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{stokes" + "_" + it_type + "}\n")
        f.write("\\label{tab:my_table}\n")
        f.write("\\end{table}\n")

        f.write("\n")

def extract_component(dataset, op_type, n_kernels, kernel):
    
    n_p = dataset.shape[0]
    
    if op_type == "Ax":
        col = 4
    elif op_type == "S":
        col = 6
    elif op_type == "Gmres":
        col = 13

    data = []

    for d in dataset:
        n_l = int(d.shape[0] / n_kernels)

        if op_type == "Gmres":
            dofs = np.array(d[kernel * n_l : kernel * n_l + n_l, 2])
            time = np.array(d[kernel * n_l : kernel * n_l + n_l, col])
            data.append(time / dofs)
        else:
            data.append(d[kernel * n_l : kernel * n_l + n_l, col])
    
    data = np.array(data, dtype=object)
    
    return data
