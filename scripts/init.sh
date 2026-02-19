#!/bin/bash
# Run this script under the root directory of GARO

# bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
pip install latex2sympy2_extended math_verify sseclient-py

wandb login 91d4dd141982939b26ac90318eefa3ae8740643a
hf auth login --token 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



echo "开始执行节点同步逻辑"
echo "RANK${RANK}" > ${PRIMUS_OUTPUT_DIR}/pre_common_ip_${PRIMUS_JOB_SUBMIT_TIMESTAMP}_${RANK}.txt

counter=`cat ${PRIMUS_OUTPUT_DIR}/pre_common_ip_${PRIMUS_JOB_SUBMIT_TIMESTAMP}_*.txt | wc -l `
while [ $counter -lt ${NNODES} ]
do
    echo "Wait for all nodes to be ready, current counter: ${counter}, all node: ${NNODES}"
    sleep 5
    counter=`cat ${PRIMUS_OUTPUT_DIR}/pre_common_ip_${PRIMUS_JOB_SUBMIT_TIMESTAMP}_*.txt | wc -l`
done



export RAY_PORT=6379
source scripts/ray_init.sh
init_ray