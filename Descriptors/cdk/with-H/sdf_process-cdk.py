import os
import numpy as np
import re
import subprocess
import pandas as pd
import rdkit 
from rdkit import Chem 
from rdkit.Chem import AllChem
from main import set_sdf_for_calc 
from utils_sdf import preprocessing_sdf
    
def compile_java(cdk_jar_path):

    cmd = ["javac","-classpath",cdk_jar_path,"AtomicDescriptorCalculator.java","AtomInfo.java"]
    subprocess.run(cmd, check=True)
    print("compiled")
    
def run_atomic_descriptor_calculator(multi_sdfs, multi_sdfs_out_cdk, cdk_jar, class_name="AtomicDescriptorCalculator"):

    os.makedirs(multi_sdfs_out_cdk, exist_ok=True)

    for filename in os.listdir(multi_sdfs):
        if filename.endswith(".sdf"):
            basename = os.path.splitext(filename)[0]
            infile = os.path.join(multi_sdfs, filename)

            subdir = os.path.join(multi_sdfs_out_cdk, basename)
            os.makedirs(subdir, exist_ok=True)

            outfile_sdf = os.path.join(subdir, f"{basename}_out.sdf")
            outfile_csv = os.path.join(subdir, f"{basename}_out.csv")

            cmd = ["java","-classpath",f".;{cdk_jar}",class_name,infile,outfile_sdf,outfile_csv]

            # print(f"calcolo molecola {filename} → {outfile_csv}")
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Errore per la molecola {filename}")
                print(f"Output: {e.stdout}")
                print(f"Errore: {e.stderr}")
                error_files.append(filename)

    print("Finito")
    
    if error_files:
        print(f"Molecole con errore: {', '.join(error_files)}")



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
                    print(f"⚠️ Errore nella lettura di {file_path}: {e}")

    print(f"\nTrovati {len(file_nan)} file .csv con almeno un valore NaN.\n")

    if file_nan:
        pd.DataFrame(file_nan).to_csv(csv_cdk_out, index=False, sep=',')
        print(f"CSV finale salvato in: {csv_cdk_out}")
    else:
        print("Nessun valore NaN trovato.")


if __name__ == "__main__":
    sdf_m = r"multisdf_file\test.sdf"
    multi_sdfs = r"path_contain_all_Sdf-files"
    cdk_jar_file = r"path containing the \cdk-2.2.jar"
    multi_sdfs_out_cdk = r"path_contain_all_Sdf-files_output_cdk_calculations"
    csv_cdk_out = r'any path to save the csv file'
    
    preprocessing_sdf(sdf_m, multi_sdfs, cln=True)
    
    compile_java(cdk_jar_file)
    run_atomic_descriptor_calculator(multi_sdfs, multi_sdfs_out_cdk, cdk_jar)
    
    check_missing_values(multi_sdfs_out_cdk, csv_cdk_out)
    
    



# instruction to run the cdk code: 
# 
# CDK-2.2.jar , AtomicDescriptorCalculator.java,  AtomInfo.java  
# should be present in the folder to run the cdk calculation 
# 
# 
