config="checkpoints/config_narrow_but_deep.yaml"
n_runs=20000
output="results/throughputs_new.csv"

matmul_precision='medium'

cmd_prefix="python throughput.py \
    --config ${config} \
    --n-runs ${n_runs} \
    --output ${output} \
    --matmul-precision ${matmul_precision}"

# == GPU timing ================================================================
device="cuda"
gpu_id="0"
cmd_cuda="${cmd_prefix} --device ${device} --gpu-id ${gpu_id}"

for batch_size in 1 2 4 8 16 32 64 128 256 384 512 768 1024 1536 2048
do
    cmd="${cmd_cuda} --batch-size ${batch_size}"
    $cmd

    cmd="${cmd_cuda} --batch-size ${batch_size} --on-gpu"
    $cmd
done

# == CPU timing ================================================================
for batch_size in 1 2 4 8 16 32 64 128 256 384 512 768 1024
do
    cmd="${cmd_prefix} --device cpu --batch-size ${batch_size}"
    $cmd
done
