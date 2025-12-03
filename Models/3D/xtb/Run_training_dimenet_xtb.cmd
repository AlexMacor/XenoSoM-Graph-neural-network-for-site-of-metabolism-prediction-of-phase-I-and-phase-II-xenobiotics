@echo off
echo Iniziato il training per le reti dimenet con i descrittori cdk_xtb 

call "path_to_env\Scripts\activate.bat"

REM --root_path e' il percorso che contiene le cartelle con all'interno i file sdf 
REM --xtb_path e' il percorso che contiene le cartelle con all'interno i file csv di output del calcolo xtb 

python Dealylation.py --root_path "Dealkylation" --xtb_path "\percorso\descrittori\xtb" > log_Dealylation_script_xtb.txt 2>&1
python Equilibrium.py --root_path "Equilibrium" --xtb_path "\percorso\descrittori\xtb" > log_Equilibrium_script_xtb.txt 2>&1
python Glucuronidation.py --root_path "Glucuronidation" --xtb_path "\percorso\descrittori\xtb" > log_Glucuronidation_script_xtb.txt 2>&1
python GlutathioneConjugation.py --root_path "GlutathioneConjugation" --xtb_path "\percorso\descrittori\xtb" > log_Glutathione_script_xtb.txt 2>&1
python Hydrolysis.py --root_path "Hydrolysis" --xtb_path "\percorso\descrittori\xtb" > log_Hydrolysis_script_xtb.txt 2>&1
python Oxidation.py --root_path "Oxidation" --xtb_path "\percorso\descrittori\xtb" > log_Oxidation_script_xtb.txt 2>&1
python Reduction.py --root_path "Reduction" --xtb_path "\percorso\descrittori\xtb" > log_Reduction_script_xtb.txt 2>&1
python Sulfonation.py --root_path "Sulfonation" --xtb_path "\percorso\descrittori\xtb" > log_Sulfonation_script_xtb.txt 2>&1