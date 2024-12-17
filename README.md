
# VASP Tools  

## Overview  
**VASP Tools** is a repository dedicated to providing a collection of Python scripts and utilities for post-processing results from **Density Functional Theory (DFT)** calculations performed using **VASP (Vienna Ab-initio Simulation Package)**. These scripts are designed to streamline and automate common tasks, making it easier to analyze, manipulate, and visualize data from VASP simulations.  

## Features  
- **Surface Generation**: Automate the creation of slabs and vacuum layers for surface calculations.  
- **Data Extraction**: Parse and analyze outputs like `OUTCAR`, `CONTCAR`, and `DOSCAR`.  
- **Visualization**: Generate plots for Density of States (DOS), Band Structures, and more.  
- **Customization**: Easily adapt scripts for specific workflows or projects.  

## Requirements  
To use the scripts in this repository, you will need:  
- **Python 3.8+**  
- Essential Python libraries, such as:  
  - `numpy`  
  - `matplotlib`  
  - `ase`  
  - `pandas`  
- **VASP Output Files**: Scripts are tailored for standard VASP output formats (`POSCAR`, `CONTCAR`, `OUTCAR`, etc.).  

Install the required Python packages using:  
```bash
pip install numpy matplotlib ase pandas
```  

## How to Use  
1. Clone the repository to your local machine:  
   ```bash
   git clone https://github.com/your-username/VASP_tools.git
   ```  

2. Navigate to the desired script's directory and run the script with appropriate input files:  
   ```bash
   python script_name.py
   ```  

3. Each script comes with detailed comments and instructions for usage. For more specific instructions, refer to the script-level documentation.  

## Scripts List  
- **`center_vacum.py`**: Create a vaccum thickness and put slab at the center of simulation box.
- **`magmom_imput.py`**: Generate magnetic moments from the last ionic stage of OUTCAR.  
   

More scripts will be added over time to expand functionality!  

## Contributions  
Contributions are welcome! If you have a script that could benefit the VASP community, feel free to submit a pull request. Alternatively, you can open an issue for feature requests or bug reports.  

### To Contribute:  
1. Fork this repository.  
2. Create a new branch:  
   ```bash
   git checkout -b feature-name
   ```  
3. Commit your changes and push:  
   ```bash
   git commit -m "Add feature description"
   git push origin feature-name
   ```  
4. Submit a pull request.  

## License  
This repository is licensed under the MIT License, allowing for free use, modification, and distribution. Refer to the `LICENSE` file for more details.  

## Acknowledgments  
Special thanks to the VASP and DFT user community for providing valuable insights and fostering collaboration.  
