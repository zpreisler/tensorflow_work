#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
def main():
    import tensorflow as tf
    from hive import hive
    from pprint import pprint

    gc_data=hive('data/gc_*.conf')
    p_data=hive('data/p_*.conf')


    with tf.Session() as session:
        summary=tf.summary.FileWriter("summary",session.graph)
        summary.close()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
