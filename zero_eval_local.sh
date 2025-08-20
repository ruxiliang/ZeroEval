#!/bin/bash


HF_HUB_ENABLE_HF_TRANSFER=1

# Initialize default values
DATA_NAME=""
model_name=""
model_pretty_name=""
n_shards=0
run_name="default"
TEMP=0
TOP_P=1.0
rp=1.0
engine_name="vllm"
batch_size=4
dp_size=1
tp_size=1
n_samples=1
TOP_K=20
disable_thinking=false

gpu_memory_utilization=0.9

MAX_TOKENS=4096; 

# Convert long options to short options
args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-name)
      args+=("-d" "$2")
      shift 2
      ;;
    --model-name)
      args+=("-m" "$2")
      shift 2
      ;;
    --model-pretty-name)
      args+=("-p" "$2")
      shift 2
      ;;
    --n-shards)
      args+=("-s" "$2")
      shift 2
      ;;
    --run-name)
      args+=("-r" "$2")
      shift 2
      ;;
    --temperature)
      args+=("-t" "$2")
      shift 2
      ;;
    --top-p)
      args+=("-o" "$2")
      shift 2
      ;;
    --repetition-penalty)
      args+=("-w" "$2")
      shift 2
      ;;
    --engine-name)
      args+=("-f" "$2")
      shift 2
      ;;
    --batch-size)
      args+=("-b" "$2")
      shift 2
      ;;
    --max-tokens)
      args+=("-x" "$2")
      shift 2
      ;;
    --dp-size)
      args+=("-e" "$2")
      shift 2
      ;;
    --tp-size)
      args+=("-g" "$2")
      shift 2
      ;;
    --top-k)
      args+=("-a" "$2")
      shift 2
      ;;
    --n-samples)
      args+=("-n" "$2")
      shift 2
      ;;
    --disable-thinking)
      disable_thinking=true
      shift
      ;;
    -*)
      args+=("$1")
      if [[ -n $2 && ! $2 =~ ^- ]]; then
        args+=("$2")
        shift 2
      else
        shift
      fi
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

# Reset positional parameters
set -- "${args[@]}"

# Parse named arguments
while getopts ":d:m:p:s:r:t:o:e:f:b:x:w:g:n:a:" opt; do
  case $opt in
    e) dp_size="$OPTARG"
    ;;
    g) tp_size="$OPTARG"
    ;;
    a) top_k="$OPTARG"
    ;;
    n) n_samples="$OPTARG"
    ;;
    d) DATA_NAME="$OPTARG"
    ;;
    m) model_name="$OPTARG"
    ;;
    p) model_pretty_name="$OPTARG"
    ;;
    s) n_shards="$OPTARG"
    ;;
    r) run_name="$OPTARG"
    ;;
    t) TEMP="$OPTARG"
    ;;
    o) TOP_P="$OPTARG"
    ;;
    w) rp="$OPTARG"
    ;;
    f) engine_name="$OPTARG"
    ;;
    b) batch_size="$OPTARG"
    ;;
    x) MAX_TOKENS="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# Echo all parsed parameters for verification
echo "=== Parsed Parameters ==="
echo "DATA_NAME: $DATA_NAME"
echo "model_name: $model_name"
echo "model_pretty_name: $model_pretty_name"
echo "n_shards: $n_shards"
echo "run_name: $run_name"
echo "TEMP: $TEMP"
echo "TOP_P: $TOP_P"
echo "rp: $rp"
echo "engine_name: $engine_name"
echo "batch_size: $batch_size"
echo "MAX_TOKENS: $MAX_TOKENS"
echo "dp_size: $dp_size"
echo "tp_size: $tp_size"
echo "top_k: $top_k"
echo "n_samples: $n_samples"
echo "disable_thinking: $disable_thinking"
echo "========================="

# Check if required arguments are provided
if [ -z "$DATA_NAME" ] || [ -z "$model_name" ] || [ -z "$model_pretty_name" ] || [ -z "$n_shards" ]; then
  echo "Usage: $0 -d DATA_NAME -m model_name -p model_pretty_name -s n_shards [-r run_name] [-t TEMP] [-o TOP_P] [-w rp] [-b batch_size]"
  exit 1
fi



CACHE_DIR=${HF_HOME:-"default"}
if [ "$run_name" = "default" ]; then
    output_dir="result_dirs/${DATA_NAME}/" 
else
    output_dir="result_dirs/${DATA_NAME}/${run_name}/" 
fi

# if model name contains "gemma-2" then use a different vllm infer backend
if [[ $model_name == *"gemma-2"* ]]; then
    export VLLM_ATTENTION_BACKEND=FLASHINFER
    # if 27b in model name, then use 0.8 gpu memory utilization
    if [[ $model_name == *"27b"* ]]; then
        gpu_memory_utilization=0.8
        echo "Using 0.8 gpu memory utilization"
    fi 
fi




max_model_len=-1

# if model name contains "phi-3.5" then use a different gpu_memory_utilization
if [[ $model_name == *"Phi-3.5"* ]]; then
    gpu_memory_utilization=0.9
    max_model_len=4096
fi

if [[ $model_name == *"70B"* ]]; then
        gpu_memory_utilization=0.9
        max_model_len=4096
fi 

echo "Using ${gpu_memory_utilization} gpu memory utilization and max_model_len=${max_model_len}"

# Helper function to add disable_thinking flag if enabled
get_disable_thinking_flag() {
    if [ "$disable_thinking" = true ]; then
        echo "\\"$'\n'"        --disable_thinking"
    else
        echo ""
    fi
}

# If the n_shards is 1, then we can directly run the model
# else, use  Data-parallellism
if [ $n_shards -eq 1 ]; then
    # gpu="0,1,2,3"; num_gpus=4; # change the number of gpus to your preference
    # echo "n_shards = 1"
    num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    # gpu= # from 0 to the last gpu id
    gpu=$(seq -s, 0 $((num_gpus - 1)))

    echo "n_shards = 1; num_gpus = $num_gpus; gpu = $gpu"
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --engine $engine_name \
        --data_name $DATA_NAME \
        --model_name $model_name \
        --run_name $run_name \
        --gpu_memory_utilization $gpu_memory_utilization \
        --max_model_len $max_model_len \
        --use_hf_conv_template --use_imend_stop \
        --download_dir $CACHE_DIR \
        --tensor_parallel_size $num_gpus \
        --dtype bfloat16 \
        --model_pretty_name $model_pretty_name \
        --top_p $TOP_P --temperature $TEMP \
        --repetition_penalty $rp \
        --batch_size $batch_size --max_tokens $MAX_TOKENS \
        --output_folder $output_dir/$(get_disable_thinking_flag)  

elif [ $n_shards -gt 1 ]; then
    echo "Using Data-parallelism"
    start_gpu=0
    num_gpus=1 
    shards_dir="${output_dir}/tmp_${model_pretty_name}"
    for ((shard_id = 0, gpu = $start_gpu; shard_id < $n_shards; shard_id++, gpu++)); do
        CUDA_VISIBLE_DEVICES=$gpu \
        python src/unified_infer.py \
            --engine $engine_name \
            --num_shards $n_shards \
            --shard_id $shard_id \
            --data_name $DATA_NAME \
            --model_name $model_name \
            --run_name $run_name \
            --gpu_memory_utilization $gpu_memory_utilization \
            --max_model_len $max_model_len \
            --use_hf_conv_template --use_imend_stop \
            --download_dir $CACHE_DIR \
            --tensor_parallel_size $num_gpus \
            --dtype bfloat16 \
            --model_pretty_name $model_pretty_name \
            --top_p $TOP_P --temperature $TEMP \
            --repetition_penalty $rp \
            --batch_size $batch_size --max_tokens $MAX_TOKENS \
            --output_folder $shards_dir/$(get_disable_thinking_flag) \
              &
    done 
    wait 
    python src/merge_results.py $shards_dir/ $model_pretty_name
    cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json

# since sglang itself supports dp and tp, we can specify it in the args
elif [ $n_shards -eq 0 ]; then
    # gpu="0,1,2,3"; num_gpus=4; # change the number of gpus to your preference
    # echo "n_shards = 1"
    num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    # gpu= # from 0 to the last gpu id
    gpu=$(seq -s, 0 $((num_gpus - 1)))

    echo "n_shards = 0; num_gpus = $num_gpus; gpu = $gpu; dp_size = $dp_size; tp_size = $tp_size"
    CUDA_VISIBLE_DEVICES=$gpu \
    python src/unified_infer.py \
        --engine $engine_name \
        --data_name $DATA_NAME \
        --model_name $model_name \
        --run_name $run_name \
        --gpu_memory_utilization $gpu_memory_utilization \
        --max_model_len $max_model_len \
        --use_hf_conv_template --use_imend_stop \
        --download_dir $CACHE_DIR \
        --data_parallel_size $dp_size \
        --tensor_parallel_size $tp_size \
        --num_outputs $n_samples \
        --dtype bfloat16 \
        --model_pretty_name $model_pretty_name \
        --top_p $TOP_P --temperature $TEMP --top_k $TOP_K --num_outputs $n_samples \
        --repetition_penalty $rp \
        --batch_size $batch_size --max_tokens $MAX_TOKENS \
        --output_folder $output_dir/$(get_disable_thinking_flag)  
else
    echo "Invalid n_shards"
    exit
fi
 