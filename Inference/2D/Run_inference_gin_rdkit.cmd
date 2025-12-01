@echo off
echo Running ..

call "\Scripts\activate.bat"

python Dealylation.py > log_Dealylation_script.txt 2>&1
python Glucuronidation.py > log_Glucuronidation_script.txt 2>&1
python GlutathioneConjugation.py > log_GlutathioneConjugation_script.txt 2>&1
python Hydrolysis.py > log_Hydrolysis_script.txt 2>&1
python Oxidation.py > log_Oxidation_script.txt 2>&1
python Reduction.py > log_Reduction_script.txt 2>&1
python Sulfonation.py > log_Sulfonation_script.txt 2>&1

