@echo off
echo Training dimenet for GNN with rdkit

python Dealylation.py --root_path "Dealkylation" > log_Dealylation_script.txt 2>&1
python Glucuronidation.py --root_path "Glucuronidation" > log_Glucuronidation_script.txt 2>&1
python GlutathioneConjugation.py --root_path "GlutathioneConjugation" > log_Glutathione_script.txt 2>&1
python Hydrolysis.py --root_path "Hydrolysis" > log_Hydrolysis_script.txt 2>&1
python Oxidation.py --root_path "Oxidation" > log_Oxidation_script.txt 2>&1
python Reduction.py --root_path "Reduction" > log_Reduction_script.txt 2>&1
python Sulfonation.py --root_path "Sulfonation" > log_Sulfonation_script.txt 2>&1
