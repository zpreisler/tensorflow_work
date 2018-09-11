#!/usr/bin/env python
from hive import hive

a=hive('data/*.conf')

from myutils import configuration
from glob import glob
f=glob('data/*.conf')
c=configuration(['data/a_-0.222_2.188.conf'])

from pprint import pprint
pprint(c.conf)
