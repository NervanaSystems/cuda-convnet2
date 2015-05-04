#! /bin/bash
# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.  All rights reserved.
# ----------------------------------------------------------------------------
logdir=/usr/local/data/datasets/imagenet/alex
fname=$1
filepath=$logdir/$fname
expname=`echo $fname | perl -ple "s|train-micro-(.*).log|\1|"`
echo $expname,,,

grep -C1 Test $filepath |  grep logprob | perl -ple "s/^logprob:  (.*)/\1 G/"  | perl -pe "s/[^G]\n/ /" | perl -ple "s/.*\((.*)%\).../ \1 /" | awk '{print $1,$3,$5,$6, $8}' | perl -ple "s/(\d),? (\d)/\1,\2/g"
