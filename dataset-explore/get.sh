#!/bin/bash

DIR=../datasets

get_conala () {
    aria2c http://www.phontron.com/download/conala-corpus-v1.1.zip -d ./ --continue=true
    unzip ./conala-corpus-v1.1.zip
    rm -f ./conala-corpus-v1.1.zip
}

get_naps () {
    curl -L -o naps.zip "https://www.dropbox.com/sh/6sp7kcfmmqjryz5/AAAp2y-wbahMpnunYROusNxza?dl=1"
    unzip naps.zip
    mkdir naps && tar xzvf naps.1.0.tar.gz -C ./naps
    rm -f *.zip *.tar.gz
}

pushd . && mkdir -p $DIR && cd $DIR

for d in "$@"
do
    echo "Getting $d"
    get_$d
    echo
done

popd
