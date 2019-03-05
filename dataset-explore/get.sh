#!/bin/bash

echo Getting CoNaLa
aria2c http://www.phontron.com/download/conala-corpus-v1.1.zip -q -d ../datasets/ --continue=true
unzip ../datasets/conala-corpus-v1.1.zip -d ../datasets/
rm -f ../datasets/conala-corpus-v1.1.zip
