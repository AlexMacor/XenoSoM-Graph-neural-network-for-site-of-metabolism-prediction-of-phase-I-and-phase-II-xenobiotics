@echo off
echo Running ..

python sdf_process-xtb.py > sdf_process-xtb_log.txt 2>&1
python get_xtb_descriptors.py > get_xtb_descriptors_log.txt 2>&1
