@echo off
echo Training dimenet with cdk and xtb 

python Dealylation.py --root_path "Dealkylation" --xtb_path "\descriptors\xtb" > log_Dealylation_script_xtb.txt 2>&1
python Glucuronidation.py --root_path "Glucuronidation" --xtb_path "\descriptors\xtb" > log_Glucuronidation_script_xtb.txt 2>&1
python GlutathioneConjugation.py --root_path "GlutathioneConjugation" --xtb_path "\descriptors\xtb" > log_Glutathione_script_xtb.txt 2>&1
python Hydrolysis.py --root_path "Hydrolysis" --xtb_path "\descriptors\xtb" > log_Hydrolysis_script_xtb.txt 2>&1
python Oxidation.py --root_path "Oxidation" --xtb_path "\descriptors\xtb" > log_Oxidation_script_xtb.txt 2>&1
python Reduction.py --root_path "Reduction" --xtb_path "\descriptors\xtb" > log_Reduction_script_xtb.txt 2>&1
python Sulfonation.py --root_path "Sulfonation" --xtb_path "\descriptors\xtb" > log_Sulfonation_script_xtb.txt 2>&1
