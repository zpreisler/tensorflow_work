#!/usr/bin/env python
from numpy import linspace
from os import system

epsilon=linspace(0.000001,2,30)
pressure=linspace(0.000001,0.2,20)
mu=linspace(-3,2,30)

i=14671
count=10000

for u in mu:
    for e in epsilon:
        i+=1137
        count+=1
        system("patchy2d -b 10 10  -N 16 --npatch 3 --nppc 1 -L 1  -g 1 -l 1.2 --mu %lf -w 8 -e %lf  -s 10000 -m 10 --pmod 10  --lambda  0 -n gc_%d --snapshot 0 --seed %d"%(u,e,count,1324+i))


#count=400000
#for p in pressure:
#    for e in epsilon:
#        i+=1137
#        count+=1
#        system("patchy2d -b 12 12  -N 20 --npatch 3 --nppc 1 -L 1  -g 0 -l 1.2  -w 8 -e %lf --npt 1 -p %lf  -s 10000 -m 10 --pmod 10  --lambda  0 -n p_%d --snapshot 0 --seed %d"%(p,e,count,1324+i))
