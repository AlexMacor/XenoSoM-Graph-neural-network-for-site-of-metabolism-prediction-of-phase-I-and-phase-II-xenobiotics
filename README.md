# Leveraging Graph Neural Networks for Joint Prediction of Sites of Metabolism and Corresponding biotransformations

Requirements
* Python 3.13.0
* Numpy 2.3.1
* pandas 2.3.0
* Matplotlib 3.10.3
* rdkit 2025.3.3
* scikit-learn 1.7.0
* torch 2.7.1+cu118
* torch-geometric 2.6.1

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
which contains the following three .py scripts

utils_folder.py: create folders  
```bash
python utils_folder.py --csv \path_to_csv_file\ --out \path_to_generate_folder\
```

main.py: data preprocessing and prepares the dependent variable y 
```bash
python main.py --csv path_to_csv_file --base main_path_to_build_the_dataset --sdf path_containing_all_the_sdf_files --out_som path_output_for_y_som
```

utils_sdf.py: gets a set of the sdf molecules for the next step
```bash
python utils_sdf.py --base_path \main_path_to_build_the_dataset\ --sdf_m \path_to_sdf_file_multisdfs\ --multi_sdfs \output_folder_for_multisdfs\
```
    
## 3 Descriptors calculation
Here you can calculate two sets of atomic descriptors based on the chemistry development toolkit (cdk) and Semiempirical Extended Tight-Binding Program Package (xtb). 

  xtb
  - 3D descriptors: before calculating these descriptors, for each molecule, a geometry optimization is              performed at the nearest minimum of the potential energy surface (PES). Next, a single-point calculation 		 was carried out.. 

    Run the following command to calculate the xtb descriptors:
    ```bash
    xtb_calculation.sh
    ```
  
    which contain the two following .py scripts: 
    sdf_process_xtb.py: perform the entire geometry optimization
  
    ```bash
    python sdf_process_xtb.py \
    --multi_sdfs /mnt/c/Users//opt/sdf \
    --input_cdk_xtb /mnt/c/Users//opt/AC-output \
    --charges_csv /mnt/c/Users//opt/charges.csv \
    --in_path /mnt/c/Users//opt/xtb/1 \
    --temp_path /mnt/c/Users//opt/xtb/2 \
    --fin_path /mnt/c/Users//opt/xtb/3 \
    --sh_file_path /mnt/c/Users//opt/xtb/chg-xtb-run-opt.sh \
    --SP_bash_script /mnt/c/Users//opt/xtb/chg-xtb-run-SP.sh
     ```

    get_xtb_descriptors.py: extract the xtb descriptors
  
    ```bash
    python script_xtb_desc.py --path_to_xtb_desc /mnt/c/Users//xtb_desc
    ```

   cdk
  - Calculations can be carried out for both 2D and 3D structures
    
    Run the following command to calculate the cdk descriptors:
    ```bash
    Cdk_calculation.cmd
    ```   
    which contains:
    
    sdf_process-cdk: takes the list of SDF files previously obtained and calculates the descriptors.
    
    ```bash
     python sdf_process-cdk.py --multi_sdfs C:\mol\split_sdf --cdk_jar_file C:\cdk\cdk-2.2.jar --  multi_sdfs_out_cdk C:\output\cdk_results --csv_cdk_out C:\output\nan_report.csv

    ```       

## 4 GNNs training 
To perform the GNN training run the following script:

```bash
Run_training.cmd
```
which performs GNN training in batch mode for each of the reaction modelled, for example, the following command was used for the glucuronidation reaction: 

```bash
python Glucuronidation.py --root_path Glucuronidation > log_Glucuronidation_script.txt 2>&1
```

## 5 Inference 
To perform inference with the previous trained model run in batch mode the following script:

```bash
Run_inference_gin_rdkit.cmd
```
which performs inference in batch mode for all the reactions using the GIN architecture with the RDKit atomic descriptors. For example, the following command was used for the glucuronidation reaction: 

```bash
python Glucuronidation.py --root_path "main folder" --file_sdf_inf_p "sdf folder inference"
```

## Usage
Please read the notes.txt file for more information about this work. 

