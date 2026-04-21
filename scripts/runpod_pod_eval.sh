#!/usr/bin/env bash
set -uo pipefail

APP_DIR="${POD_EVAL_APP_DIR:-/opt/nlsh}"
WORKSPACE_DIR="${POD_EVAL_WORKSPACE:-/workspace}"
ARTIFACT_DIR="${POD_EVAL_OUTPUT_DIR:-${WORKSPACE_DIR}/nlsh-artifacts}"
DATASET="${POD_EVAL_DATASET:-${APP_DIR}/data/dev.messages.jsonl}"
MANIFEST="${POD_EVAL_MANIFEST:-${APP_DIR}/configs/pod_eval_models.json}"
PYTHON_BIN="${POD_EVAL_PYTHON:-python}"

export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export NLSH_VLLM_LIKE="${NLSH_VLLM_LIKE:-1}"

mkdir -p "${HF_HOME}" "${ARTIFACT_DIR}"

STARTUP_LOG="${ARTIFACT_DIR}/startup.log"
exec > >(tee -a "${STARTUP_LOG}") 2>&1

echo "nlsh Runpod pod eval startup"
echo "app_dir=${APP_DIR}"
echo "workspace_dir=${WORKSPACE_DIR}"
echo "hf_home=${HF_HOME}"
echo "artifact_dir=${ARTIFACT_DIR}"
echo "dataset=${DATASET}"
echo "manifest=${MANIFEST}"

status=0

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required. Set it from a Runpod secret before launching the pod."
  status=2
else
  cd "${APP_DIR}" || status=$?
fi

if [[ "${status}" -eq 0 ]]; then
  "${PYTHON_BIN}" scripts/pod_eval.py --manifest "${MANIFEST}" download-models
  status=$?
fi

if [[ "${status}" -eq 0 ]]; then
  suite_args=(
    --manifest "${MANIFEST}"
    run-suite
    --dataset "${DATASET}"
    --output-dir "${ARTIFACT_DIR}"
  )

  if [[ -n "${POD_EVAL_LIMIT:-}" ]]; then
    suite_args+=(--limit "${POD_EVAL_LIMIT}")
  fi
  if [[ -n "${POD_EVAL_TIMEOUT:-}" ]]; then
    suite_args+=(--timeout "${POD_EVAL_TIMEOUT}")
  fi
  if [[ -n "${POD_EVAL_STARTUP_TIMEOUT:-}" ]]; then
    suite_args+=(--startup-timeout "${POD_EVAL_STARTUP_TIMEOUT}")
  fi

  "${PYTHON_BIN}" scripts/pod_eval.py "${suite_args[@]}"
  status=$?
fi

echo "${status}" > "${ARTIFACT_DIR}/last_exit_code"
echo "pod eval finished with exit code ${status}"

if [[ "${POD_EVAL_EXIT_AFTER:-0}" == "1" ]]; then
  exit "${status}"
fi

echo "keeping container alive for inspection; set POD_EVAL_EXIT_AFTER=1 to exit after the batch"
tail -f /dev/null
