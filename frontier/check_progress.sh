job_root=$1
for folder in "$job_root"/mlp*
do
    echo -e "$(head -n1 $folder/checkpoints/last_saved_epoch) $folder"
done
