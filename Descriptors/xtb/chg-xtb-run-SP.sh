#!/bin/bash

in_path='/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/1' # folder number 1 
xtb_path='/mnt/c/Users/Alessio Macorano/Desktop/Database-metabolismo-glucuronidazione/xtb'
temp_path='/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/2' # dato che questo script viene richiamato 
																					  # da xtb-run-loop-freq output dir e' uguale a file_path dell'altro script 

mkdir -p "$temp_path"
options="--gfn 2 --vfukui --esp --alpb water"

declare -A charges
charges["molecola_1767"]=0
charges["molecola_19966"]=0
charges["molecola_22114"]=0
charges["molecola_25865"]=0
charges["molecola_27631"]=0
charges["molecola_33064"]=0
charges["molecola_41291"]=0
charges["molecola_41467"]=0
charges["molecola_44234"]=0

for input_file in "$in_path"/*.sdf; do
  file_name=$(basename "$input_file")
  file_prefix="${file_name%.sdf}"

  if [[ -z "${charges[$file_prefix]+x}" ]]; then
    echo "Nessuna carica definita per $file_prefix. Salto."
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
    echo "✅ Finished $file_name, output: $output_file"
    
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
