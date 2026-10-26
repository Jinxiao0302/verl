#!/usr/bin/env bash
# 只启动推理（standalone rollout server）做精度验证，不起 trainer/FSDP/ref。
# 基于 run_kimi_k3_mindspeed_w4a8_vs_bf16.sh 改写：
#   - 入口换成 verl.trainer.main_generation_mm（main_generation_server 的多模态变体，
#     会把 parquet 里 images 列的图片以 image_url 形式发给 server，并按 \boxed{} 对
#     reward_model.ground_truth 统计准确率）；
#   - W4A8 默认走 refit（方案 B）：load_format=dummy 启动后，由驱动把 checkpoint 的
#     bf16 权重经 RL 同款 update_weights_from_ipc/quant_weights 在线量化推进 server
#     （checkpoint 里是高精权重 + 量化描述文件，vLLM 直接加载不会在线量化）；
#   - BF16 对照组设 ROLLOUT_QUANTIZATION=none（自动改为真实加载，不做 refit）；
#   - 去掉了 free_cache_engine / sleep mode 等仅 RL 训练需要的项；
#   - 默认 temperature=0、n=1（贪心、可复现），可用环境变量覆盖。
set -euo pipefail

PROJECT_ROOT="/mnt/share/z00931161/kimi-k3/fsdpturbo-backend/"
start_time="${VERL_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

nnodes=1
npus_per_node=8
visible_devices_default=0,1,2,3,4,5,6,7
rollout_tp_size=8
rollout_ep_size=8
max_response_length="${VERL_MAX_RESPONSE_LENGTH:-512}"
rollout_max_num_seqs="${VERL_VLLM_MAX_NUM_SEQS:-32}"
rollout_max_num_batched_tokens="${VERL_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
rollout_enforce_eager="${VERL_VLLM_ENFORCE_EAGER:-False}"

# 精度验证默认贪心采样；要模拟训练分布可 GEN_TEMPERATURE=1.0 GEN_N=4
gen_temperature="${GEN_TEMPERATURE:-0.0}"
gen_n="${GEN_N:-1}"

model_path="${MODEL_PATH:-/mnt/share/z00804445/verl_fsdp_turbo/kimi_k3/model/K3-top16-pruned_bf16_countbench_128_n4_12layers-W4A8-20261015}"
# BF16 对照组：
# model_path="${MODEL_PATH:-/mnt/share/z00931161/weights/K3-top16-pruned_bf16_countbench_128_n4_12layers}"
rollout_quantization="${ROLLOUT_QUANTIZATION:-ascend}"   # BF16 对照时设为 none（下方会跳过该配置）
exp_name="${VERL_EXPERIMENT_NAME:-kimi_k3_gen_${start_time}}"

gen_file="${GEN_FILE:-/mnt/share/w00848461/datasets/countbenchqa_lite/train_448.parquet}"
gen_output="${GEN_OUTPUT:-/mnt/share/z00804445/verl_fsdp_turbo/kimi_k3/verl/gen_results/${exp_name}.parquet}"

fsdp_turbo_path="/mnt/share/z00804445/verl_fsdp_turbo/kimi_k3/FSDPTurbo"
vllm_path="${PROJECT_ROOT}/vllm"
vllm_ascend_path="${PROJECT_ROOT}/vllm-ascend"
runtime_site="${PROJECT_ROOT}/.runtime-site"
cann_python_site=/mnt/share/j00949742/conda_envs/cann/9.2.0b030/cann-9.2.0/python/site-packages

# The checkpoint's Python files are executable remote code. Keep actor and
# rollout on the reviewed FSDP-Turbo implementation for every launch.
cp -f "${fsdp_turbo_path}/fsdp_turbo/models/kimi/"*.py "${model_path}/"

export PYTHONPATH="${runtime_site}:${vllm_path}:${vllm_ascend_path}:${fsdp_turbo_path}:${fsdp_turbo_path}/examples:${cann_python_site}"
export FSDP_TURBO_ROOT="${fsdp_turbo_path}"
export VLLM_ASCEND_PATH="${vllm_ascend_path}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${visible_devices_default}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NON_MEGATRON="${NON_MEGATRON:-true}"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARN}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-WARN}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ASCEND_ENABLE_FLASHCOMM="${VLLM_ASCEND_ENABLE_FLASHCOMM:-1}"
export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MULTI_STREAM_MEMORY_REUSE="${MULTI_STREAM_MEMORY_REUSE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-data0.3001}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-data0.3001}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-50000}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-17340}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-256}"
export P2P_HCCL_BUFFSIZE="${P2P_HCCL_BUFFSIZE:-20}"
export VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
export VERL_FSDP_TURBO_VISION_ATTN_IMPLEMENTATION="${VERL_FSDP_TURBO_VISION_ATTN_IMPLEMENTATION:-flash_attention_2}"
export KIMI_TRAIN_USE_PARTITION_INVARIANT_KDA_FORWARD="${KIMI_TRAIN_USE_PARTITION_INVARIANT_KDA_FORWARD:-1}"

local_ray_owned=0
cleanup_local_ray() {
    if (( local_ray_owned == 1 )); then
        echo "Stopping local Ray runtime"
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup_local_ray EXIT INT TERM

ray stop --force >/dev/null 2>&1 || true
node_ip="${NODE_IP:-172.27.13.95}"
ray start \
    --head \
    --port="${RAY_PORT:-6792}" \
    --node-ip-address="${node_ip}" \
    --dashboard-host="${node_ip}" \
    --dashboard-port="${RAY_DASHBOARD_PORT:-8262}" \
    --num-cpus="${RAY_NUM_CPUS:-64}" \
    --object-store-memory="${RAY_OBJECT_STORE_MEMORY:-68719476736}" \
    --resources="{\"NPU\": ${npus_per_node}}"
local_ray_owned=1

gen_log_dir="${GEN_LOG_DIR:-/mnt/share/z00804445/verl_fsdp_turbo/kimi_k3/verl/gen_results}"
mkdir -p "${gen_log_dir}"
gen_log="${gen_log_dir}/${exp_name}.log"

max_prompt_length=1024
rollout_multimodal_token_margin="${VERL_VLLM_MULTIMODAL_TOKEN_MARGIN:-16384}"
rollout_max_model_len="${VERL_VLLM_MAX_MODEL_LEN:-$((max_prompt_length + max_response_length + rollout_multimodal_token_margin))}"

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=data
    actor_rollout_ref.rollout.expert_parallel_size=${rollout_ep_size}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.6}
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.n=${gen_n}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager}
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.response_length=${max_response_length}
    actor_rollout_ref.rollout.temperature=${gen_temperature}
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
)

# W4A8 量化模型需要；BF16 对照组用 ROLLOUT_QUANTIZATION=none 跳过
if [[ "${rollout_quantization,,}" != "none" ]]; then
    ROLLOUT_CONFIG+=(
        actor_rollout_ref.rollout.quantization=${rollout_quantization}
    )
fi

# refit（方案 B）：以 load_format=dummy 启动（不读权重，秒级起来），随后由
# 驱动从 checkpoint 的 bf16 safetensors 读权重，走 RL 同款的
# update_weights_from_ipc -> quant_weights(npu_dynamic_mx_quant) 在线量化并加载。
# W4A8 必须开（vLLM 不会在线量化 checkpoint 里的高精权重）；BF16 可直接真实加载。
# 注意：refit 传输走共享内存，单 tensor 最大约 2.4GB，bucket 不能小于它；
# 8 个 sender 并发各占一个 bucket 的 /dev/shm（4096MB x 8 = 32GB），shm 不足时
# 调小 REFIT_BUCKET_MB（但必须 > 2600）。
refit_from_checkpoint="${REFIT_FROM_CHECKPOINT:-1}"
if [[ "${rollout_quantization,,}" == "none" ]]; then
    refit_from_checkpoint="${REFIT_FROM_CHECKPOINT:-0}"
fi
if [[ "${refit_from_checkpoint}" == "1" ]]; then
    ROLLOUT_CONFIG+=(
        actor_rollout_ref.rollout.load_format=dummy
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${REFIT_BUCKET_MB:-4096}
    )
else
    # RolloutConfig 的 load_format 默认就是 dummy（会触发 refit），
    # 真实加载必须显式给 auto
    ROLLOUT_CONFIG+=(
        actor_rollout_ref.rollout.load_format=auto
    )
fi

# 与训练脚本一致：非 eager 时开启 decode cudagraph
case "${rollout_enforce_eager,,}" in
    false|0|no|off)
        ROLLOUT_CONFIG+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=FULL_DECODE_ONLY
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[1,2,4,8,16]'
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=16
        )
        ;;
esac

echo "Kimi gen: model=${model_path} quantization=${rollout_quantization} refit=${refit_from_checkpoint} temperature=${gen_temperature} n=${gen_n}"
echo "Kimi gen: data=${gen_file} output=${gen_output}"
echo "Kimi gen log: ${gen_log}"

python3 -m verl.trainer.main_generation_mm \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${npus_per_node} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.trust_remote_code=True \
    data.train_files="${gen_file}" \
    +data.output_path="${gen_output}" \
    data.prompt_key=prompt \
    data.trust_remote_code=True \
    "${ROLLOUT_CONFIG[@]}" \
    "$@" 2>&1 | tee "${gen_log}"
