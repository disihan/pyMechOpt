# pyMechOpt: A Python toolbox for optimizing of reaction mechanisms

pyMechOpt can optimize the chemical reaction mechanism and reduce the difference with the detailed reaction mechanism.

## Installation

It is recommended to install pyMechOpt in a newly created python environment.

| Package                                             | version  |
| --------------------------------------------------- | -------- |
| [Cantera](https://cantera.org/)                     | \>=3.00  |
| [Pymoo](https://pymoo.org/)                         | \>=0.6.0 |
| [pyDecision](https://github.com/Valdecy/pyDecision) | \>=4.5.8 |

After installing the above package, you can follow the following command to install pyMechOpt:

    git clone https://github.com/disihan/pyMechOpt.git
    pip install ./pyMechOpt

## Usage

You can import modules by using the following code:

    import numpy as np
    import cantera as ct

    from pyMechOpt.multi_obj import mo_problem      # For multi-objective
    # from pyMechOpt.single_obj import so_problem   # For single-objective
    # from pyMechOpt.line_search import ls_problem  # For line search method

Firstly, provide the mechanism that needs to be optimized and an accurate mechanism:

    mech_detailed = 'mech/gri30.yaml'               # detailed mech
    mech_reduced = 'mech/gri30-r45.yaml'            # reduced mech

Given the species to be optimized and the initial conditions:

    spcs_int = ['CH4',
                'O2']
    spcs_peak = ['CH2O',
                'CO',
                'CO2']

    pressure = 1e05 * np.array([1, 1, 1, 100, 100, 100])
    temperature = np.array([1000, 1000, 1000, 1000, 1000, 1000])
    ratio = np.array([0.25, 1, 4, 0.25, 1, 4])

    fuel = "CH4:1"
    oxydizer = "O2:1"

Then you can start optimizing.

    opt_prob = mo_problem(
        mech_detailed,
        mech_reduced,
        fuel,
        oxydizer,
        temperature,
        ratio,
        pressure,
        spcs_int,
        spcs_peak,
        range_input=0.1,
        weight=[1, 1, 1, 1],
        hist_dir="./hist_nsga3/",
        res_dir="./res_nsga3/",
    )
    opt_prob.calc_skip()
    opt_prob.run(algorithm="NSGA3", max_gen=400, pop_size=200)

## Cite us

[pyMechOpt: A Python toolbox for optimizing of reaction mechanisms](https://doi.org/10.1016/j.softx.2024.102001)

Bibtex:

> @article{pyMechOpt,  
> &nbsp; &nbsp; &nbsp; &nbsp; title = {pyMechOpt: A Python toolbox for optimizing of reaction mechanisms},  
> &nbsp; &nbsp; &nbsp; &nbsp; journal = {SoftwareX},  
> &nbsp; &nbsp; &nbsp; &nbsp; volume = {29},  
> &nbsp; &nbsp; &nbsp; &nbsp; pages = {102001},  
> &nbsp; &nbsp; &nbsp; &nbsp; year = {2025},  
> &nbsp; &nbsp; &nbsp; &nbsp; issn = {2352-7110},  
> &nbsp; &nbsp; &nbsp; &nbsp; doi = {https://doi.org/10.1016/j.softx.2024.102001},  
> &nbsp; &nbsp; &nbsp; &nbsp; url = {https://www.sciencedirect.com/science/article/pii/S2352711024003716},  
> &nbsp; &nbsp; &nbsp; &nbsp; author = {Sihan Di and Nanjia Yu and Shutao Han and Haodong He},  
> &nbsp; &nbsp; &nbsp; &nbsp; keywords = {Reaction mechanisms, Optimization, Python toolbox},  
> }
