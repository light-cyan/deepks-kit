nohup python -u -m deepks iterate args.yaml slurm.yaml >> log.iter 2> err.iter &
echo $! > PID
