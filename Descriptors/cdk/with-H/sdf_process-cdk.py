import os
import numpy as np
import re
import subprocess
import pandas as pd
import rdkit 
from rdkit import Chem 
from rdkit.Chem import AllChem
import argparse
    
def compile_java(cdk_jar_file):

    cmd = ["javac","-classpath",cdk_jar_file,"AtomicDescriptorCalculator.java","AtomInfo.java"]
    subprocess.run(cmd, check=True)
    print("compiled")
    
def run_atomic_descriptor_calculator(multi_sdfs, multi_sdfs_out_cdk, cdk_jar_file, class_name="AtomicDescriptorCalculator"):

    os.makedirs(multi_sdfs_out_cdk, exist_ok=True)
    error_files = []
    for filename in os.listdir(multi_sdfs):
        if filename.endswith(".sdf"):
            basename = os.path.splitext(filename)[0]
            infile = os.path.join(multi_sdfs, filename)

            subdir = os.path.join(multi_sdfs_out_cdk, basename)
            os.makedirs(subdir, exist_ok=True)

            outfile_sdf = os.path.join(subdir, f"{basename}_out.sdf")
            outfile_csv = os.path.join(subdir, f"{basename}_out.csv")

            cmd = ["java","-classpath",f".;{cdk_jar_file}",class_name,infile,outfile_sdf,outfile_csv]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                
                error_files.append(filename)

    print("Finish")
    
    if error_files:
        print(f"Molecules with errors: {', '.join(error_files)}")



# check eventually missing values in the cdk output 

def check_missing_values(multi_sdfs_out_cdk, csv_cdk_out):
    file_nan = []

    for root, _, files in os.walk(multi_sdfs_out_cdk):
        for f in files:
            if f.endswith('.csv'):
                file_path = os.path.join(root, f)
                try:
                    df = pd.read_csv(file_path, sep=',')
                    if df.isnull().values.any():
                        nan_colonne = df.columns[df.isnull().any()].tolist()
                        file_nan.append({"file": f, "colonne_nan": ", ".join(nan_colonne)})
                except Exception as e:
                    print(f" Error during reading {file_path}: {e}")

    print(f"\nFound {len(file_nan)} file .csv with at least one NaN.\n")

    if file_nan:
        pd.DataFrame(file_nan).to_csv(csv_cdk_out, index=False, sep=',')
        print(f"Final CSV saved in: {csv_cdk_out}")
    else:
        print("No NaN values found.")


if __name__ == "__main__":
    #multi_sdfs = r"path_contain_all_Sdf-files"
    #cdk_jar_file = r"path containing the \cdk-2.2.jar"
    #multi_sdfs_out_cdk = r"path_contain_all_Sdf-files_output_cdk_calculations"
    #csv_cdk_out = r'any path to save the csv file'

    parser = argparse.ArgumentParser(description="cdk calculation on sdf file")
    
    parser.add_argument("--multi_sdfs", required=True, help="folder with the single sdf file")
    parser.add_argument("--cdk_jar_file", required=True, help="Path of cdk-2.2.jar")
    parser.add_argument("--multi_sdfs_out_cdk", required=True, help="folder cdk output")
    parser.add_argument("--csv_cdk_out", required=True, help="path_missing_values")

    args = parser.parse_args()

    multi_sdfs = args.multi_sdfs
    cdk_jar_file = args.cdk_jar_file
    multi_sdfs_out_cdk = args.multi_sdfs_out_cdk
    csv_cdk_out = args.csv_cdk_out
        
    compile_java(cdk_jar_file)
    run_atomic_descriptor_calculator(multi_sdfs, multi_sdfs_out_cdk, cdk_jar_file)
    
    check_missing_values(multi_sdfs_out_cdk, csv_cdk_out)
    
    



# instruction to run the cdk code: 
# 
# CDK-2.2.jar , AtomicDescriptorCalculator.java,  AtomInfo.java  
# should be present in the folder to run the cdk calculation 
# 
# 
