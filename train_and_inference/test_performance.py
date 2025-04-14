import subprocess
import textwrap

num = [1, 2, 4, 8]

for num_cpus in num:
    slurm_script = textwrap.dedent(f'''\
    #!/bin/bash

    #SBATCH --account=dukepauli
    #SBATCH --partition=gpu_high_priority
    #SBATCH --qos=user_qos_dukepauli
    #SBATCH --gres=gpu:volta:1
    #SBATCH --job-name=plantsegnet
    #SBATCH --nodes=1
    #SBATCH --ntasks={num_cpus}
    #SBATCH --time=2:00:00

    module load micromamba
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate plantsegnet
    cd ~/code/PlantSegNet/train_and_inference
    echo Running with {num_cpus} tasks...
    python predict_and_save_hpc.py -s=9 -i=10 -p=/home/u9/bhuppenthal/xdisk/vrbio/test/segmentation_pointclouds_{num_cpus}
    ''')

    with open(f'../test_{num_cpus}.slurm', 'w') as f:
        f.write(slurm_script)
    
    result = subprocess.run(['sbatch', f'../test_{num_cpus}.slurm'])