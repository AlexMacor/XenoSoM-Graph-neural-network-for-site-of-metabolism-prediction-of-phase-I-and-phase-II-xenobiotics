@echo off
echo Training for GNNs with cdk

python Dealylation.py --root_path "\Dealkylation" --descr_path "\percorso\descrittori\cdk" --use_cdk > log_Dealylation_script_cdk.txt 2>&1
python Equilibrium.py --root_path "\Equilibrium" --descr_path "\percorso\descrittori\cdk" --use_cdk > log_Equilibrium_script_cdk.txt 2>&1
python Glucuronidation.py --root_path "\Glucuronidation" --descr_path "\percorso\descrittori\cdk" --use_cdk > log_Glucuronidation_script_cdk.txt 2>&1
python GlutathioneConjugation.py --root_path "\GlutathioneConjugation" --descr_path "\percorso\descrittori\cdk" --use_cdk > log_GlutathioneConjugation_script_cdk.txt 2>&1
python Hydrolysis.py --root_path "\Hydrolysis" --descr_path "\percorso\descrittori\cdk" --use_cdk > log_Hydrolysis_script_cdk.txt 2>&1
python Oxidation.py --root_path "\Oxidation" --descr_path "\percorso\descrittori\cdk" --use_cdk > log_Oxidation_script_cdk.txt 2>&1
python Reduction.py --root_path "\Reduction" --descr_path "\percorso\descrittori\cdk" --use_cdk > log_Reduction_script_cdk.txt 2>&1
python Sulfonation.py --root_path "\Sulfonation" --descr_path "\percorso\descrittori\cdk" --use_cdk > log_Sulfonation_script_cdk.txt 2>&1

