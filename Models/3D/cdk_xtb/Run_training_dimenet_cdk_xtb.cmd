@echo off
echo Training dimenet using cdk and xtb descriptors

python Dealylation.py --root_path "Dealkylation" --cdk_path "\percorso\descrittori\cdk" --xtb_path "\percorso\descrittori\xtb" > log_Dealylation_script_cdk_xtb.txt 2>&1
python Glucuronidation.py --root_path "Glucuronidation" --cdk_path "\percorso\descrittori\cdk" --xtb_path "\percorso\descrittori\xtb" > log_Glucuronidation_script_cdk_xtb.txt 2>&1
python GlutathioneConjugation.py --root_path "GlutathioneConjugation" --cdk_path "\percorso\descrittori\cdk" --xtb_path "\percorso\descrittori\xtb" > log_Glutathione_script_cdk_xtb.txt 2>&1
python Hydrolysis.py --root_path "Hydrolysis" --cdk_path "\percorso\descrittori\cdk" --xtb_path "\percorso\descrittori\xtb" > log_Hydrolysis_script_cdk_xtb.txt 2>&1
python Oxidation.py --root_path "Oxidation" --cdk_path "\percorso\descrittori\cdk" --xtb_path "\percorso\descrittori\xtb" > log_Oxidation_script_cdk_xtb.txt 2>&1
python Reduction.py --root_path "Reduction" --cdk_path "\percorso\descrittori\cdk" --xtb_path "\percorso\descrittori\xtb" > log_Reduction_script_cdk_xtb.txt 2>&1
python Sulfonation.py --root_path "Sulfonation" --cdk_path "\percorso\descrittori\cdk" --xtb_path "\percorso\descrittori\xtb" > log_Sulfonation_script_cdk_xtb.txt 2>&1
