#!/bin/bash

lines_all=$(git ls-files | xargs cat | wc -l)
lines_doxygen=$(cat doc/Doxyfile doc/mainpage.txt | wc -l)
lines=$(echo "$lines_all - $lines_doxygen" | bc)

echo $lines
