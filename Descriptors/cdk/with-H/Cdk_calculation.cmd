@echo off
echo Running ..

python sdf_process-cdk.py --multi_sdfs "C:\molecole\split_sdf" --cdk_jar_file "C:\cdk\cdk-2.2.jar" --multi_sdfs_out_cdk "C:\output\cdk_results" --csv_cdk_out "C:\output\nan_report.csv"
