import os
import shutil
import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_folders", type=int, required=True)
args = parser.parse_args()

# xtb software only for linux 

bash_script_path = r"/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/chg-xtb-run-opt.sh" # file script with the declare of the charge 
n_folders = args.n_folders # set the number of folder identical to the number of input files. 

in_path = r"/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/1" # folder number 1 
temp_path = r"/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/2"  # folder number 2
fin_path = r"/mnt/c/Users/Alessio Macorano/Desktop/Database-test-reazione-equilibrioum/opt/xtb/3"  # folder number 3


subprocess.run(["bash", bash_script_path], check=True) 

while True:
    n_cartelle = len([d for d in os.listdir(temp_path) if os.path.isdir(os.path.join(temp_path, d))])
    print(f"Cartelle trovate: {n_cartelle}", end='\r')
    if n_cartelle >= n_folders:
        break
    time.sleep(60)

# Main cycle in order to have no imaginary frequency during the geoemtry optimization
while True:
    for file in os.listdir(in_path):
        temp_path_cl = os.path.join(in_path, file)
        if os.path.isfile(temp_path_cl):
            try:
                os.remove(temp_path_cl)
            except Exception as e:
                print(f"Errore rimuovendo {temp_path_cl}: {e}")

    for root, dirs, files in os.walk(temp_path):
        for file in files:
            if file.endswith('.out') and not file.endswith('g98.out'):
                out_path = os.path.join(root, file)
                move = False

                with open(out_path, 'r') as f:
                    for line in f:
                        if 'significant imaginary frequency' in line or 'significant imaginary frequencies' in line:
                            move = True
                            break

                if move:
                    nome_cartella = os.path.basename(root)
                    nuova_cartella = os.path.join(fin_path, nome_cartella)

                    shutil.move(root, nuova_cartella)

                    for subfile in os.listdir(nuova_cartella):
                        full_path = os.path.join(nuova_cartella, subfile)
                        if subfile.endswith('hess.sdf'):
                            new_name = subfile.replace('_xtbhess', '')
                            destinazione_finale = os.path.join(in_path, new_name)
                            try:
                                shutil.move(full_path, destinazione_finale)
                            except Exception as e:
                                print(f"Errore spostando {subfile}: {e}")
                        else:
                            try:
                                os.remove(full_path)
                            except Exception as e:
                                print(f"Errore rimuovendo {full_path}: {e}")

                    try:
                        if not os.listdir(nuova_cartella):
                            os.rmdir(nuova_cartella)
                    except Exception as e:
                        print(f"Errore rimuovendo cartella {nuova_cartella}: {e}")

                    break  # passa alla prossima cartella
                    
    # Re run the sdf files with negative IF ===
    if os.listdir(in_path):
        print("Rilancio script Bash per elaborare i nuovi .sdf...")
        try:
            subprocess.run(["bash", bash_script_path], check=True)
            print("Script Bash completato.")
        except subprocess.CalledProcessError as e:
            print(f"Errore eseguendo il Bash script: {e}")
        time.sleep(5) 
        
    
    cartelle_ok = 0
    
    for dir_name in os.listdir(temp_path):
        dir_path = os.path.join(temp_path, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        has_imaginary = False
        
        for file in os.listdir(dir_path):
            if file.endswith('.out') and not file.endswith('g98.out'):
                with open(os.path.join(dir_path, file), 'r') as f:
                    for line in f:
                        if 'significant imaginary frequency' in line or 'significant imaginary frequencies' in line:
                            has_imaginary = True
                            break
                if has_imaginary:
                    break
        if not has_imaginary:
            cartelle_ok += 1
    
    cartelle_non_ok = n_folders - cartelle_ok
    print(f"Cartelle con file sdf senza frequenze immaginarie: {cartelle_ok}/{n_folders}")
    
    if cartelle_ok >= n_folders:
        print("ok")
        cartelle_non_ok = n_folders - cartelle_ok
        print(f"Alcuni file sdf devono essere riottimizzati: {cartelle_non_ok}")
        break  # FINE
    else:
        cartelle_non_ok = n_folders - cartelle_ok
        print(f"Alcuni file sdf devono essere riottimizzati: ({cartelle_non_ok}). Ripeto...")
        time.sleep(60)
    