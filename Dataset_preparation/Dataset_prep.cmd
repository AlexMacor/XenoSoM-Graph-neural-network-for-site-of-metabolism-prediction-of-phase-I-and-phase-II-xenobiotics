@echo off
echo Running ..

python utils_folder.py > utils_folder_log.txt 2>&1
python main.py > main_log.txt 2>&1
python utils_sdf.py > utils_sdf_log.txt 2>&1

