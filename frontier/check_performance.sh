job_root=$1
for folder in "$job_root"/mlp*
do
    echo -e "$(tail -n1 $folder/checkpoints/valid_log.csv) $folder"
done
