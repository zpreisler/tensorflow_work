#!/usr/bin/env python
from numpy import linspace
from os import system


epsilon=linspace(0.1,12,20)
mu=linspace(-6,6,20)

i=1546

for e in epsilon:
    for u in mu:
        print(u,e)
        i+=1137
        system("patchy2d -b 10 10  -N 16 --npatch 4 --nppc 1 -L 1  -g 1 -l 1.2 --mu %lf -w 8 -e %lf  -s 1000000 -m 100 --pmod 100  --lambda  0 -n a_%.3lf_%.3lf --snapshot 0 --seed %d"%(u,e,u,e,1324+i))
