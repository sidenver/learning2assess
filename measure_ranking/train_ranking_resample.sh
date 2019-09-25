#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --output=jupyter_%j.log
##SBATCH --qos=batch
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=60gb
#SBATCH --time=1-00:00:00

date;hostname;pwd

# unset XDG_RUNTIME_DIR 
export XDG_RUNTIME_DIR="/fs/clip-psych/shing/"

source ~/.bash_profile

source activate tf-gpu
 
echo -e "\nStarting on the $(hostname) server."

cd /fs/clip-psych
cd /fs/clip-scratch
cd /fs/clip-psych/shing/learning2assess/measure_ranking/

# module load cuda/9.0.176
# module load cudnn/7.0.64

module load cuda/10.0.130
module load cudnn/7.5.0

input="/fs/clip-psych/shing/task_B_user_embedding_with_binary_pretrain/task_B.train.all.prediction"

for i in {1..100}
do
    outdir="out_${i}"
    echo $outdir
    seed=$RANDOM
    echo $seed
    mkdir -p $outdir

    python -u split_and_convert.py -i $input -o $outdir -s $seed -p 0.2

    for loss in pairwise_logistic_loss pairwise_hinge_loss pairwise_soft_zero_one_loss softmax_loss sigmoid_cross_entropy_loss mean_squared_loss list_mle_loss approx_ndcg_loss approx_mrr_loss
    do
        echo $loss
        python -u train_ranking_resample.py -l $loss -o $outdir -s $seed --train "${outdir}/train.svm" --test "${outdir}/dev.svm"
    done
done

date