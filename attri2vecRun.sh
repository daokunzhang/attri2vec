#!/bin/bash
./attri2vec -graph Facebook.txt -output Facebook_attri2vec1_emb.txt -time Facebook_attri2vec1_time.txt -syn Facebook_attri2vec1_syn.txt -option 1 -size 128
./attri2vec -graph Facebook.txt -output Facebook_attri2vec2_emb.txt -time Facebook_attri2vec2_time.txt -syn Facebook_attri2vec2_syn.txt -option 2 -size 128
./attri2vec -graph Facebook.txt -output Facebook_attri2vec3_emb.txt -time Facebook_attri2vec3_time.txt -syn Facebook_attri2vec3_syn.txt -option 3 -size 128
./attri2vec -graph Facebook.txt -output Facebook_attri2vec4_emb.txt -time Facebook_attri2vec4_time.txt -syn Facebook_attri2vec4_syn.txt -option 4 -size 128
