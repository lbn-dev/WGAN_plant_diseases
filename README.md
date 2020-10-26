# WGAN_plant_diseases
Experiment 1: train CNN without classic data augmentation.   
sbatch job_cnnWithoutGenerator.script  
Experiemnt 2: train CNN with classic data augmentation.  
sbatch job_cnn.script  
Experiemnt 3: train CNN with classic data augmentation and WGAN.
sbatch job_wganWithoutLSR.script  sbatch job_cnn.script  
Experiemnt 4: train CNN with classic data augmentation and WGAN-LSR.  
sbatch job_wgan.script  sbatch job_cnn.script  
