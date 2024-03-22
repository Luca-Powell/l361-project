#!/bin/bash
#SBATCH -c 10
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:0
#SBATCH --job-name=template
#SBATCH --partition=normal
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G

poetry shell

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)

# Set up the redis password
# redis_password=$(uuidgen)
# export redis_password
# # Set up the head node IP address
# ip="localhost"
# main_port=8379
# port1=8700
# port2=8701
# port3=10001
# port4=8702
# port5=10002
# port6=19999

# export NUM_GPUS=`echo $CUDA_VISIBLE_DEVICES | awk 'BEGIN{FS=","};{print NF}'`

# ip=$(hostname --ip-address) # making redis-address

# # if we detect a space character in the head node IP, we'll
# # convert it to an ipv4 address. This step is optional.
# if [[ "$ip" == *" "* ]]; then
#   IFS=' ' read -ra ADDR <<< "$ip"
#   if [[ ${#ADDR[0]} -gt 16 ]]; then
#     ip=${ADDR[1]}
#   else
#     ip=${ADDR[0]}
#   fi
#   echo "IPV6 address detected. We split the IPV4 address as $ip"
# fi

# Start Ray session
# poetry run ray start --head --node-ip-address=$ip --num-gpus=${NUM_GPUS} --num-cpus=${SLURM_CPUS_PER_TASK} \
#     --port=$main_port \
#     --node-manager-port=$port1 \
#     --object-manager-port=$port2 \
#     --ray-client-server-port=$port3 \
#     --redis-shard-ports=$port4 \
#     --min-worker-port=$port5 \
#     --max-worker-port=$port6 \
#     --redis-password="$redis_password" \
#     --verbose \
#     --include-dashboard=False \
#     --block &

# sleep 30

# poetry run ray status

# sleep 5

# Set the custom hydra arguments that will be passed to the server and the node manager
CUSTOM_HYDRA_ARGS="--config-name=cifar10 ++run_uuid=$run_uuid" # ++ray_address=auto ++ray_redis_password=$redis_password ++ray_node_ip_address=$ip"

poetry run python -m project.main $CUSTOM_HYDRA_ARGS 

# srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive bash launch_template.sh