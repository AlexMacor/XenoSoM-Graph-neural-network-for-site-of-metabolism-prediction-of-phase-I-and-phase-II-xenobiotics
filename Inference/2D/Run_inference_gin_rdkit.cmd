@echo off
echo Running ..

call "\Scripts\activate.bat"

python Dealylation.py --root_path "main folder" --file_sdf_inf_p "sdf folder inference"
python Glucuronidation.py --root_path "main folder" --file_sdf_inf_p "sdf folder inference"
python GlutathioneConjugation.py --root_path "main folder" --file_sdf_inf_p "sdf folder inference"
python Hydrolysis.py --root_path "main folder" --file_sdf_inf_p "sdf folder inference"
python Oxidation.py --root_path "main folder" --file_sdf_inf_p "sdf folder inference"
python Reduction.py --root_path "main folder" --file_sdf_inf_p "sdf folder inference"
python Sulfonation.py --root_path "main folder" --file_sdf_inf_p "sdf folder inference"
