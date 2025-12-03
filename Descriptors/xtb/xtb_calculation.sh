#!/bin/bash
echo Running xtb calculation

python3 sdf_process_xtb.py \
  --multi_sdfs "/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/sdf" \
  --input_cdk_xtb "/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/AC-output" \
  --charges_csv "/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/charges.csv" \
  --in_path "/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/1" \
  --temp_path "/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/2" \
  --fin_path "/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/3" \
  --sh_file_path "/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/chg-xtb-run-opt.sh" \
  --SP_bash_script "/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/chg-xtb-run-SP.sh"

python3 script_xtb_desc.py --path_to_xtb_desc "/mnt/c/Users/Alessio Macorano/Desktop/xtb_desc"