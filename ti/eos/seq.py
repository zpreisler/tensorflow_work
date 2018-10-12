#!/usr/bin/env python
from numpy import linspace
from os import system

p=0.1
w=6

a=linspace(1,8,192)
for i,x in enumerate(a):
    system("npt gas.conf -s 0 -n fluid{} -e {} -p {} --new_width {} ".format(i,x,p,w))
