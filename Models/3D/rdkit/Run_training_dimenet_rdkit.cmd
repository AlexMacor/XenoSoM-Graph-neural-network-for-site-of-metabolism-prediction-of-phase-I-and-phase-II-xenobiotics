@echo off
echo Iniziato il training per le reti dimenet con i descrittori rdkit

call "path_to_env\Scripts\activate.bat"

REM --root_path e' il percorso che contiene le cartelle con all'interno i file sdf 

python Dealylation.py --root_path "Dealkylation" > log_Dealylation_script.txt 2>&1
python Equilibrium.py --root_path "Equilibrium" > log_Equilibrium_script.txt 2>&1
python Glucuronidation.py --root_path "Glucuronidation" > log_Glucuronidation_script.txt 2>&1
python GlutathioneConjugation.py --root_path "GlutathioneConjugation" > log_Glutathione_script.txt 2>&1
python Hydrolysis.py --root_path "Hydrolysis" > log_Hydrolysis_script.txt 2>&1
python Oxidation.py --root_path "Oxidation" > log_Oxidation_script.txt 2>&1
python Reduction.py --root_path "Reduction" > log_Reduction_script.txt 2>&1
python Sulfonation.py --root_path "Sulfonation" > log_Sulfonation_script.txt 2>&1