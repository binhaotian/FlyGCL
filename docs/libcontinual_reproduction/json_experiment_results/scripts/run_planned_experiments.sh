#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/flygcl/bin/python}"
DEVICE="${DEVICE:-0}"
SEEDS="${SEEDS:-1 2 3 4 5}"
DRY_RUN="${DRY_RUN:-1}"
GENERATE_CONFIGS="${GENERATE_CONFIGS:-1}"
ONLY=""
GROUP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY="$2"
      shift 2
      ;;
    --group)
      GROUP="$2"
      shift 2
      ;;
    *)
      echo "未知参数：$1" >&2
      exit 2
      ;;
  esac
done

if [[ "${GENERATE_CONFIGS}" == "1" ]]; then
  "${PYTHON_BIN}" reproduce/flyprompt/generate_planned_configs.py
fi

run_or_print() {
  local name="$1"
  local group="$2"
  local config="$3"
  local results_root="$4"
  local status="${5:-ready}"

  if [[ -n "${ONLY}" && "${ONLY}" != "${name}" ]]; then
    return 0
  fi
  if [[ -n "${GROUP}" && "${GROUP}" != "${group}" ]]; then
    return 0
  fi

  echo
  local display_status="${status}"
  if [[ "${status}" == "ready" ]]; then
    display_status="可运行"
  fi
  echo "===== ${name} [实验组=${group}, 状态=${display_status}] ====="

  if [[ "${status}" != "ready" ]]; then
    echo "跳过：${status}"
    echo "计划配置：${config}"
    return 0
  fi

  for seed in ${SEEDS}; do
    local cmd=("${PYTHON_BIN}" run_trainer.py --config "${config}" --seed "${seed}" --device "${DEVICE}")
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '仅打印命令:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
    fi
  done

  local collect_cmd=("${PYTHON_BIN}" reproduce/flyprompt/collect_gcl_results.py --results-root "${results_root}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '仅打印命令:'
    printf ' %q' "${collect_cmd[@]}"
    printf '\n'
  else
    "${collect_cmd[@]}"
  fi
}

# 优先级 1：ImageNet-R / CUB-200 默认实验。运行前需确认数据目录存在。
run_or_print "imagenetr_default" "datasets" \
  "flyprompt_plan_imagenetr_sup21k_default_amp.yaml" \
  "./results_plan_imagenetr_sup21k_default_amp/log/FlyPrompt"

run_or_print "cub200_default" "datasets" \
  "flyprompt_plan_cub200_sup21k_default_amp.yaml" \
  "./results_plan_cub200_sup21k_default_amp/log/FlyPrompt"

# 优先级 2：对应 N_variants.tex 的 r_D 变体实验。
run_or_print "cifar_rd0" "n_variants" \
  "flyprompt_plan_cifar100_rd0_rb10_amp.yaml" \
  "./results_plan_cifar100_rd0_rb10_amp/log/FlyPrompt"

run_or_print "cifar_rd100" "n_variants" \
  "flyprompt_plan_cifar100_rd100_rb10_amp.yaml" \
  "./results_plan_cifar100_rd100_rb10_amp/log/FlyPrompt"

# 优先级 3：对应 M_variants.tex 的 r_B 变体实验。
run_or_print "cifar_rb30" "m_variants" \
  "flyprompt_plan_cifar100_rd50_rb30_amp.yaml" \
  "./results_plan_cifar100_rd50_rb30_amp/log/FlyPrompt"

run_or_print "cifar_rb50" "m_variants" \
  "flyprompt_plan_cifar100_rd50_rb50_amp.yaml" \
  "./results_plan_cifar100_rd50_rb50_amp/log/FlyPrompt"

# 优先级 4：组件消融实验。
run_or_print "ablation_no_ema" "ablation" \
  "flyprompt_plan_cifar100_no_ema_amp.yaml" \
  "./results_plan_cifar100_no_ema_amp/log/FlyPrompt"

run_or_print "ablation_no_prompt_no_ema" "ablation" \
  "flyprompt_plan_cifar100_no_prompt_no_ema_amp.yaml" \
  "./results_plan_cifar100_no_prompt_no_ema_amp/log/FlyPrompt"

run_or_print "ablation_no_prompt_ema" "ablation" \
  "flyprompt_plan_cifar100_no_prompt_ema_amp.yaml" \
  "./results_plan_cifar100_no_prompt_ema_amp/log/FlyPrompt"

run_or_print "ablation_no_rear_no_ema" "ablation" \
  "flyprompt_plan_cifar100_no_rear_no_ema_amp.yaml" \
  "./results_plan_cifar100_no_rear_no_ema_amp/log/FlyPrompt"

run_or_print "ablation_no_rear_ema" "ablation" \
  "flyprompt_plan_cifar100_no_rear_ema_amp.yaml" \
  "./results_plan_cifar100_no_rear_ema_amp/log/FlyPrompt"

run_or_print "ablation_ranpac_dagger" "ablation" \
  "flyprompt_plan_cifar100_ranpac_dagger_amp.yaml" \
  "./results_plan_cifar100_ranpac_dagger_amp/log/RanPACGCL"

run_or_print "ablation_no_prompt_proxy" "ablation" \
  "flyprompt_plan_cifar100_no_prompt_proxy_amp.yaml" \
  "./results_plan_cifar100_no_prompt_proxy_amp/log/FlyPrompt" \
  "历史近似配置：prompt_length=0；正式 no-prompt 请使用 ablation_no_prompt_no_ema / ablation_no_prompt_ema"

run_or_print "ablation_no_rear_proxy" "ablation" \
  "flyprompt_plan_cifar100_no_rear_proxy_amp.yaml" \
  "./results_plan_cifar100_no_rear_proxy_amp/log/FlyPrompt" \
  "历史近似配置；正式 no-REAR 请使用 ablation_no_rear_no_ema / ablation_no_rear_ema"

# 优先级 5a：ensemble sweep。当前 FlyPrompt 对未知 ensemble 方法会回退到 mean，
# 因此这里不包含 min-entropy 相关行。
run_or_print "ensemble_mean" "ensemble" \
  "flyprompt_plan_cifar100_ensemble_mean_amp.yaml" \
  "./results_plan_cifar100_ensemble_mean_amp/log/FlyPrompt"

run_or_print "ensemble_max_prob" "ensemble" \
  "flyprompt_plan_cifar100_ensemble_max_prob_amp.yaml" \
  "./results_plan_cifar100_ensemble_max_prob_amp/log/FlyPrompt"

run_or_print "ensemble_softmax_mean" "ensemble" \
  "flyprompt_plan_cifar100_ensemble_softmax_mean_amp.yaml" \
  "./results_plan_cifar100_ensemble_softmax_mean_amp/log/FlyPrompt"

# 优先级 5b：EMA sweep。
run_or_print "ema_online_only" "ema" \
  "flyprompt_plan_cifar100_ema_online_only_amp.yaml" \
  "./results_plan_cifar100_ema_online_only_amp/log/FlyPrompt"

run_or_print "ema_09" "ema" \
  "flyprompt_plan_cifar100_ema_09_amp.yaml" \
  "./results_plan_cifar100_ema_09_amp/log/FlyPrompt"

run_or_print "ema_099" "ema" \
  "flyprompt_plan_cifar100_ema_099_amp.yaml" \
  "./results_plan_cifar100_ema_099_amp/log/FlyPrompt"

run_or_print "ema_0999" "ema" \
  "flyprompt_plan_cifar100_ema_0999_amp.yaml" \
  "./results_plan_cifar100_ema_0999_amp/log/FlyPrompt"

run_or_print "ema_09_099_0999" "ema" \
  "flyprompt_plan_cifar100_ema_09_099_0999_amp.yaml" \
  "./results_plan_cifar100_ema_09_099_0999_amp/log/FlyPrompt"

# 优先级 7：mask sweep。
run_or_print "mask_seen_class" "mask" \
  "flyprompt_plan_cifar100_mask_seen_class_amp.yaml" \
  "./results_plan_cifar100_mask_seen_class_amp/log/FlyPrompt"

echo
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "完成。这次是 dry-run，没有启动训练。设置 DRY_RUN=0 才会启动选中的实验。"
else
  echo "完成。选中范围内的实验已经执行结束。"
fi
