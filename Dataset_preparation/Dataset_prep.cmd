@echo off
echo Running ..

REM there is an example of the complete_path the main.py script

python utils_folder.py --csv "\path_to_csv_file\" --out "\path_to_generate_folder\"
python main.py --csv "path_to_csv_file" --base "main_path_to_build_the_dataset" --sdf "path_containing_all_the_sdf_files" --out_som "path_output_for_y_som"
python utils_sdf.py --base_path "\main_path_to_build_the_dataset\" --sdf_m "\path_to_sdf_file_multisdfs\" --multi_sdfs "\output_folder_for_multisdfs\"
	

