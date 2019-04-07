#!/bin/bash

data_file="tranx.0.2.0.zip"
to_dir="tranx-data"

aria2c -s16 -x16 http://www.cs.cmu.edu/~pengchey/${data_file}

mkdir ${to_dir}
unzip ${data_file} -d ${to_dir}
rm ${data_file}