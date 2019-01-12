#!/bin/bash
gcc -Wall -g -c "attri2vec.c" -o attri2vec.o
g++ -o attri2vec attri2vec.o
