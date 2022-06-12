#!/bin/bash
START_HOUR=$1
END_HOUR=$((1+START_HOUR))
NUMBER_OF_OPS=6
op_file_prefix="gharrafaMatOD_"$START_HOUR"-"$END_HOUR"_R"
ip_file="gharrafaMatOD_"$START_HOUR"-"$END_HOUR"_R0"

for ((i=1; i<=$NUMBER_OF_OPS; i++)); do
  head -6 $ip_file > "$op_file_prefix$i"
done

tail -n +7 "$ip_file" |
{
while IFS= read -r line
do
  set -- $line
  for ((i=1; i<=$NUMBER_OF_OPS; i++))
  do
    value=`python -c "from math import ceil; from numpy.random import uniform; print(ceil(uniform(0.5, 1.5) * $3))"`
    echo -e "$1\t$2\t$value" >> "$op_file_prefix$i"
  done
done
}
echo "Done generating random matOD files"
