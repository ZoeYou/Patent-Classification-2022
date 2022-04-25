#!/bin/bash
#SBATCH --job-name=attentionxml    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=10       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --partition=almanach          # Name of the partition
#SBATCH --gres=gpu:1     # GPU nodes are only available in gpu partition
#SBATCH --mem=128G                # Total memory allocated
#SBATCH --hint=multithread       # we get logical cores (threads) not physical (cores)
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=res%A_%a.out   # output file name
#SBATCH --error=res%A_%a.out    # error file name

echo "### Running $SLURM_JOB_NAME ###"

cd ${SLURM_SUBMIT_DIR}

module purge


# Set your conda environment
source /home/$USER/.bashrc
source activate py37

bash f_${SLURM_ARRAY_TASK_ID}.in

#python preprocess.py \
#--text-path data/INPI_fr_title_desc_2020_4/train_raw_texts.txt \
#--tokenized-path data/INPI_fr_title_desc_2020_4/train_texts.txt \
#--label-path data/INPI_fr_title_desc_2020_4/train_labels.txt \
#--vocab-path data/INPI_fr_title_desc_2020_4/vocab.npy \
#--emb-path data/INPI_fr_title_desc_2020_4/emb_init.npy \
#--w2v-model data/wiki.fr.gensim

#python preprocess.py \
#--text-path data/INPI_fr_title_desc_2020_4/test_raw_texts.txt \
#--tokenized-path data/INPI_fr_title_desc_2020_4/test_texts.txt \
#--label-path data/INPI_fr_title_desc_2020_4/test_labels.txt \
#--vocab-path data/INPI_fr_title_desc_2020_4/vocab.npy 
