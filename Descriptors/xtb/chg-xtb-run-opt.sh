#!/bin/bash

in_path='/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/1' 
xtb_path='/mnt/c/Users/Alessio Macorano/Desktop/Database-metabolismo-glucuronidazione/xtb'
temp_path='/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/2' 
													 

mkdir -p "$temp_path"
options="--ohess tight --alpb water"

declare -A charges

for input_file in "$in_path"/*.sdf; do
  file_name=$(basename "$input_file")
  file_prefix="${file_name%.sdf}"

  if [[ -z "${charges[$file_prefix]+x}" ]]; then
    echo "No defined charge for $file_prefix. Skip"
    continue
  fi

  charge=${charges[$file_prefix]}

  molecule_dir="$temp_path/$file_prefix"
  mkdir -p "$molecule_dir"
  cp "$input_file" "$molecule_dir/"

  output_file="$molecule_dir/${file_prefix}.out"
  cd "$molecule_dir" || exit

  echo "Processing $file_name with charge $charge..."

  "$xtb_path" "$file_name" $options --charge "$charge" > "$output_file"

  if [[ $? -eq 0 ]]; then
    echo "Finished $file_name, output: $output_file"
    
    for generated_file in *; do
      if [[ $generated_file == *.cosmo || $generated_file == *.dat || 
            $generated_file == charges* || $generated_file == wbo* || 
            $generated_file == *opt.sdf || $generated_file == *opt.log || 
            $generated_file == *xtbhess.sdf ]]; then

        new_name="${file_prefix}_${generated_file}"
        mv "$generated_file" "$new_name"
        echo "Renamed $generated_file → $new_name"
      fi
    done
  else
    echo "Errore su $file_name"
  fi

  cd - > /dev/null || exit
done
