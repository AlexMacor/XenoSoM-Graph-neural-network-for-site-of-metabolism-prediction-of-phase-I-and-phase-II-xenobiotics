#!/bin/bash
echo Running xtb calculation

python3 sdf_process_xtb.py \
  --multi_sdfs "/mnt/c/Users//opt/sdf" \
  --input_cdk_xtb "/mnt/c/Users//opt/AC-output" \
  --charges_csv "/mnt/c/Users//opt/charges.csv" \
  --in_path "/mnt/c/Users//opt/xtb/1" \
  --temp_path "/mnt/c/Users//opt/xtb/2" \
  --fin_path "/mnt/c/Users//opt/xtb/3" \
  --sh_file_path "/mnt/c/Users//opt/xtb/chg-xtb-run-opt.sh" \
  --SP_bash_script "/mnt/c/Users//opt/xtb/chg-xtb-run-SP.sh"

python3 script_xtb_desc.py --path_to_xtb_desc "/mnt/c/Users//xtb_desc"
