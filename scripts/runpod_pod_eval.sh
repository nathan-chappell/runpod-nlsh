#!/usr/bin/env bash
set -uo pipefail

APP_DIR="${POD_EVAL_APP_DIR:-/opt/nlsh}"
WORKSPACE_DIR="${POD_EVAL_WORKSPACE:-/workspace}"
ARTIFACT_DIR="${POD_EVAL_OUTPUT_DIR:-${WORKSPACE_DIR}/nlsh-artifacts}"
DATASET="${POD_EVAL_DATASET:-${APP_DIR}/data/dev.messages.jsonl}"
MANIFEST="${POD_EVAL_MANIFEST:-${APP_DIR}/configs/pod_eval_models.json}"
BOOTSTRAP_PYTHON="${POD_EVAL_BOOTSTRAP_PYTHON:-python}"
PYTHON_BIN="${POD_EVAL_PYTHON:-python}"
VENV_DIR="${POD_EVAL_VENV:-${WORKSPACE_DIR}/nlsh-venv}"
VLLM_SPEC="${POD_EVAL_VLLM_SPEC:-vllm}"

export HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/hf-cache}"
export TMPDIR="${TMPDIR:-${WORKSPACE_DIR}/tmp}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORKSPACE_DIR}/triton-cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${WORKSPACE_DIR}/vllm-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORKSPACE_DIR}/.cache}"
export NLSH_VLLM_LIKE="${NLSH_VLLM_LIKE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

mkdir -p "${HF_HOME}" "${ARTIFACT_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${XDG_CACHE_HOME}"

STARTUP_LOG="${ARTIFACT_DIR}/startup.log"
exec > >(tee -a "${STARTUP_LOG}") 2>&1

echo "nlsh Runpod pod eval startup"
echo "app_dir=${APP_DIR}"
echo "workspace_dir=${WORKSPACE_DIR}"
echo "hf_home=${HF_HOME}"
echo "tmpdir=${TMPDIR}"
echo "triton_cache_dir=${TRITON_CACHE_DIR}"
echo "vllm_cache_root=${VLLM_CACHE_ROOT}"
echo "artifact_dir=${ARTIFACT_DIR}"
echo "dataset=${DATASET}"
echo "manifest=${MANIFEST}"
echo "venv_dir=${VENV_DIR}"
echo "vllm_spec=${VLLM_SPEC}"
echo "VLLM_USE_V1=${VLLM_USE_V1}"
echo "VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD}"
echo "CC=${CC}"
echo "CXX=${CXX}"

status=0
runpod_services_pid=""

if [[ "${POD_EVAL_START_RUNPOD_SERVICES:-1}" == "1" && -x /start.sh ]]; then
  echo "starting Runpod base services with /start.sh"
  /start.sh &
  runpod_services_pid=$!
  sleep "${POD_EVAL_RUNPOD_SERVICE_DELAY:-2}"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is required. Set it from a Runpod secret before launching the pod."
  status=2
else
  cd "${APP_DIR}" || status=$?
fi

if [[ "${status}" -eq 0 && "${POD_EVAL_SKIP_VENV:-0}" != "1" ]]; then
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "creating persistent Python environment at ${VENV_DIR}"
    "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
    status=$?
  fi

  if [[ "${status}" -eq 0 ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
    "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
    status=$?
  fi

  if [[ "${status}" -eq 0 ]]; then
    "${PYTHON_BIN}" -m pip install -e "${APP_DIR}"
    status=$?
  fi

  if [[ "${status}" -eq 0 ]]; then
    if "${PYTHON_BIN}" -c "import vllm" >/dev/null 2>&1; then
      echo "vLLM is already available in ${VENV_DIR}"
    else
      echo "installing ${VLLM_SPEC} into ${VENV_DIR}"
      "${PYTHON_BIN}" -m pip install "${VLLM_SPEC}"
      status=$?
    fi
  fi
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
  if [[ -n "${POD_EVAL_VLLM_ARGS:-}" ]]; then
    read -r -a extra_vllm_args <<< "${POD_EVAL_VLLM_ARGS}"
    for extra_vllm_arg in "${extra_vllm_args[@]}"; do
      suite_args+=(--vllm-arg "${extra_vllm_arg}")
    done
  fi

  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
  "${PYTHON_BIN}" - <<'PY' || true
import importlib.metadata
for package in ("torch", "vllm"):
    try:
        print(f"{package}={importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}=not-installed")
PY

  "${PYTHON_BIN}" scripts/pod_eval.py "${suite_args[@]}"
  status=$?
fi

echo "${status}" > "${ARTIFACT_DIR}/last_exit_code"
echo "pod eval finished with exit code ${status}"

if [[ "${POD_EVAL_EXIT_AFTER:-0}" == "1" ]]; then
  exit "${status}"
fi

echo "keeping container alive for inspection; set POD_EVAL_EXIT_AFTER=1 to exit after the batch"
if [[ -n "${runpod_services_pid}" ]]; then
  if ! kill -0 "${runpod_services_pid}" >/dev/null 2>&1; then
    wait "${runpod_services_pid}" || true
  fi
fi
tail -f /dev/null
