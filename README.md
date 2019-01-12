# attri2vec

Code for DMKD paper "Attributed Network Embedding via Subspace Discovery"

Authors: Daokun Zhang, Jie Yin, Xingquan Zhu and Chengqi Zhang

Contact: Daokun Zhang (daokunzhang2015@gmail.com)

Please run the "attri2vecRun.sh" file to run this implementation on the Facebook network.

The format of the input network is as following:

In the input file, the first line is the number of nodes ("node_num") and the number of node content attributes ("feat_num") seprated by whitespace:

    node_num feat_num

From the second line, one by one, each user's id, neighbor list and profile features are provided. The following is the information for a node:

    node_id
    neigh_num
    neigh_id1 neigh_id2 neigh_id3 ......
    nonzero_feat_num
    nonzero_attribute_id1 nonzero_attribute_val1 nonzero_attribute_id2 nonzero_attribute_val2 ......

Above, "node_id" is the id of current node. "neigh_num" is the number of neighbors of current node, and "neigh_id1, neigh_id2, neigh_id3..." is the neighbor list of current node. "nonzero_feat_num" is the number of observed nonzero feature values of the current node and "nonzero_attribute_id1 nonzero_attribute_val1 nonzero_attribute_id2 nonzero_attribute_val2 ......" is the nonzero feature value list of current node, where "nonzero_attribute_id1" is the id of the observed attribute in which current node take nonzero value and "nonzero_attribute_val1" is the corresponding attribute value.

Taking the "Facebook.graph.txt" as an example, its input is:

    4039 1403
    0
    347
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347
    27
    64 1 83 1 324 1 396 1 418 1 585 1 664 1 766 1 927 1 948 1 1044 1 1046 1 1049 1 1083 1 1166 1 1169 1 1172 1 1174 1 1214 1 1273 1 1276 1 1346 1 1348 1 1349 1 1350 1 1354 1 1358 1
    1
    17
    0 48 53 54 73 88 92 119 126 133 194 236 280 299 315 322 346
    2
    663 1 927 1
    2
    10
    0 20 115 116 149 226 312 326 333 343
    8
    318 1 573 1 664 1 758 1 764 1 813 1 926 1 954 1
    3
    17
    0 9 25 26 67 72 85 122 142 170 188 200 228 274 280 283 323
    15
    26 1 83 1 307 1 396 1 575 1 581 1 664 1 758 1 927 1 956 1 1177 1 1179 1 1218 1 1349 1 1364 1
    4
    10
    0 78 152 181 195 218 273 275 306 328
    4
    396 1 552 1 664 1 927 1
    ......

The options of attri2vec are as follows:

	-graph <file>
		The input <file> for network embedding
	-output <file>
		Use <file> to save the resulting network embeddings
	-time <file>
		Use <file> to save running time
	-syn <file>
		Use <file> to save the weights used for constructing node embeddings from node attributes
	-option <int>
		The mapping option for constructing embedding from attributes; with 1 for Linear mapping, 2 for ReLU mapping, 3 for Fourier mapping, 4 for Sigmoid mapping; default is 1
	-size <int>
		Set size of learned dimensions; default is 128
	-window <int>
		Window size for collecting node context pairs; default is 10
	-walknum <int>
		The number of random walks starting from per node; default is 40
	-walklen <int>
		The length of random walks; default is 100
	-negative <int>
		Number of negative examples; default is 5, common values are 3 - 10
	-alpha <float>
		Set the starting learning rate; default is 0.025
	-samples <int>
		Set the number of iterations for stochastic gradient descent as <int> Million; default is 100




