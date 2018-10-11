#!/bin/bash

#SBATCH --job-name=eos6
#SBATCH --account=etna
#SBATCH --ntasks=192
#SBATCH --partition=etna
#SBATCH --time=48:0:0
#SBATCH --signal=B:10@300

trap 'kill -s 2 $(jobs -pr); wait' INT KILL TERM

for i in `ls fluid*.conf`
do
   a[count]=$i
   let count++
done

let count--
for i in $(seq 0 1 $count)
do
   echo $i ${a[i]}
done

for ((myproc=0; myproc < $SLURM_NPROCS; myproc++))
do
   npt ${a[myproc]} -s 100000 -o 1 -v 0 --pmod 50 -m 50 --snapshot 0 > log_${a[myproc]} &
done

wait
