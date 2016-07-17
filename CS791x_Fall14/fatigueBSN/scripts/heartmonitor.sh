#!/usr/bin/env bash

date +%s%N
echo "BEGIN"
gatttool -b 00:22:D0:3E:4B:04 -I
echo "END"
date +%s%N
