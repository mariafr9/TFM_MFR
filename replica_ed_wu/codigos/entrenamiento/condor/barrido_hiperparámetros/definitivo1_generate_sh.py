import subprocess

project_path = '/lhome/ext/i3m121/i3m1214/replica_ed_wu/codigos/entrenamiento/condor'
condor_path = '/barrido_hiperparámetros/'
shNew_path = 'normalizacion/'
txt_file = 'definitivo1.txt'
bash_line = '#!/bin/bash'

work_direction = f"cd {project_path}"
condorDir_command = f"cd {project_path}{condor_path}"
newFolder_command = f"mkdir {project_path}{condor_path}{shNew_path}"
newDir_command = f"cd {shNew_path}"
setup_command  = f"source {project_path}{condor_path}setup.sh"

subprocess.run(newFolder_command, shell=True)
print(newFolder_command)

txt_path = f"{project_path}{condor_path}{txt_file}"
project_name = 'normalizacion_study'
with open(txt_path, 'r') as file:
    for line_num, line in enumerate(file, start=1):
        file_name = f"{project_path}{condor_path}{shNew_path}{project_name}_{line_num}.sh"
        with open(file_name, 'w') as sh_file:
            sh_file.write(bash_line)
            sh_file.write("\n")
            #sh_file.write(work_direction)
            sh_file.write(condorDir_command)
            sh_file.write("\n")
            sh_file.write(setup_command)
            sh_file.write("\n")
            sh_file.write(line)
            barra = "█" * line_num 
            print(f"\r[{barra}]",  end='', flush=True)
print('')
