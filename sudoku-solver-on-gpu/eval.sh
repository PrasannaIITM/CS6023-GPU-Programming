%%writefile eval.sh
current_dir=$(pwd)
INPUT="input"
OUTPUT="output_bt"

nvcc obt.cu -arch=sm_37

max=100
i=1
start=`date +%s.%N`
while [ "$i" -le "$max" ]
do
   input_file="$INPUT/input$i.txt"
   echo "$input_file"
   output_file="$OUTPUT/output$i.txt"
   echo "$output_file"
   ./a.out "$input_file" "$output_file" 8192 16
   i=$(( i + 1 ))
done
end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo "Time taken : $runtime"

EOP="output"
OP="output_bt"

i=1
ans=0
while [ "$i" -le "$max" ]
do
   expectedOP_file="$EOP/output$i.txt"
  
   OP_file="$OP/output$i.txt"
   
   diff -w "$expectedOP_file" "$OP_file" > /dev/null 2>&1
   exit_code=$?
   if [ $exit_code -eq 0 ]
   then
     ans=$(( ans + 1 ))
   fi
   
   i=$(( i + 1 ))
done

echo "Number of successes : $ans"
