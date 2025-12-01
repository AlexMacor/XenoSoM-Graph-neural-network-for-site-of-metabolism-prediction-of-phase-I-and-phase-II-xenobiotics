@echo off
echo Running ..

python sdf_process-cdk.py > sdf_process-cdk_log.txt 2>&1
python get_cdk_descriptors.py > get_cdk_descriptors_log.txt 2>&1
