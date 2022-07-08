current_dir=$(pwd)
INPUT="input"
OUTPUT="outputCPU"

filename="cpu.c"

gcc cpu.c -lm

max=100
i=1
start=`date +%s.%N`
while [ "$i" -le "$max" ]
do
   input_file="$INPUT/input$i.txt"
   echo "$input_file"
   output_file="$OUTPUT/output$i.txt"
   echo "$output_file"
   ./a.out "$input_file" "$output_file"
   i=$(( i + 1 ))
done
end=`date +%s.%N`

runtime=$( echo "$end - $start" | bc -l )
echo "Time taken : $runtime"

