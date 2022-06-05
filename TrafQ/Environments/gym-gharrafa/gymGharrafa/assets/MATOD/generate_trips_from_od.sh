TAZ_FILE="../TAZ/gharrafa_taz.xml"
START_HOUR=$1
END_HOUR=$((1+START_HOUR))
OD_FILE_PREFIX="gharrafaMatOD_"$START_HOUR"-"$END_HOUR"_R"
OP_FILE_PREFIX="gha_trips_"$START_HOUR"-"$END_HOUR"_R"
OP_FILE_SUFFIX=".rou.xml"

for i in {0..6}
do
  od_file=$OD_FILE_PREFIX$i
  op_file=$OP_FILE_PREFIX$i$OP_FILE_SUFFIX
  echo "Generating routes in $op_file"
  od2trips -n $TAZ_FILE -d $od_file -o $op_file
done
