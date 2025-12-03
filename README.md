# GraphSoM_EX-a-site-of-metabolism-predictor-for-phase-I-and-phase-II-metabolic-reaction

Requipments
* Python 3.13.0
* Numpy 2.3.1
* pandas 2.3.0
* Matplotlib 3.10.3
* rdkit 2025.3.3
* scikit-learn 1.7.0
* torch                   2.7.1+cu118
* torch-geometric         2.6.1

## 1. Create and activate the virtual environment

Run the following command in your project directory:
```bash
python3 -m venv .GNN_SoM_env
```

## 2. Activate the virtual environment

On Linux: source .GNN_SoM_env/bin/activate
On Windows: .GNN_SoM_env\Scripts\activate

## 3 Dataset preparation
Dataset preparation involves several steps. Starting from the CSV file, the entire data structure is created in order to perform both descriptor calculations and GNN training/inference.

Run the following command in your project directory:
```bash
Dataset_prep.cmd
```
which contain the three following .py scripts





This .cmd program will run three Python scripts in batch mode. You should open it with any text editor and modify the paths as required. Details about the three scripts are provided in the notes.txt file.

## 3 Descriptors calculation
Here you can calculate two set of atomic descriptors based on the chemistry development toolkit (cdk) and Semiempirical Extended Tight-Binding Program Package (xtb). 

  xtb
  - 3D descriptors: before calculating these descriptors, for each molecule, a geometry optimization is              performed at the nearest minimum of the potential energy surface (PES). Next a single point using was            carried out. 

    Run the following command to calculate the xtb descriptors:
    ```bash
    xtb_calculation.sh
    ```
  
    which contain the two following .py scripts: 
    sdf_process_xtb.py: perform the entire geometry optimization
  
    ```bash
    python sdf_process_xtb.py \
    --multi_sdfs "/mnt/c/Users//opt/sdf" \
    --input_cdk_xtb "/mnt/c/Users//opt/AC-output" \
    --charges_csv "/mnt/c/Users//opt/charges.csv" \
    --in_path "/mnt/c/Users//opt/xtb/1" \
    --temp_path "/mnt/c/Users//opt/xtb/2" \
    --fin_path "/mnt/c/Users//opt/xtb/3" \
    --sh_file_path "/mnt/c/Users//opt/xtb/chg-xtb-run-opt.sh" \
    --SP_bash_script "/mnt/c/Users//opt/xtb/chg-xtb-run-SP.sh"
     ```

    get_xtb_descriptors.py: extract the xtb descriptors
  
    ```bash
    python3 script_xtb_desc.py --path_to_xtb_desc "/mnt/c/Users//xtb_desc"
    ```





   cdk
  - 3D descriptors: before calculating these descriptors, for each molecule, a geometry optimization is              performed at the nearest minimum of the potential energy surface (PES). Next a single point using was            carried out.  

Run the following command in your project directory:
```bash
Dataset_prep.cmd
```
This .cmd program will run three Python scripts in batch mode. You should open it with any text editor and modify the paths as required. Details about the three scripts are provided in the notes.txt file.




## Usage
Read the notes.txt file carefully for a detailed explanation from dataset preparation to inference.

