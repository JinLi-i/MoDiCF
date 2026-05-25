#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
LOCAL_SOURCE="${LOCAL_SOURCE:-${ROOT_DIR}/../modicf_author_data_download}"
MODICF_DATA_URL="${MODICF_DATA_URL:-}"
DATASETS=("$@")

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  DATASETS=(baby tiktok allrecipes)
fi

copy_dataset() {
  local source_root="$1"
  local dataset="$2"
  local source_dir="${source_root}/${dataset}"
  local target_dir="${DATA_DIR}/${dataset}"

  if [[ ! -d "${source_dir}" ]]; then
    echo "Missing dataset directory: ${source_dir}" >&2
    return 1
  fi

  mkdir -p "${target_dir}"
  cp -a "${source_dir}/." "${target_dir}/"
  echo "Installed ${dataset} into ${target_dir}"
}

install_from_local_source() {
  local source_root="$1"

  for dataset in "${DATASETS[@]}"; do
    copy_dataset "${source_root}" "${dataset}"
  done
}

install_from_archive_url() {
  local url="$1"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT

  if command -v gdown >/dev/null 2>&1 && [[ "${url}" == *drive.google.com/drive/folders/* ]]; then
    gdown --folder "${url}" -O "${tmp_dir}"
  else
    local archive="${tmp_dir}/modicf_data"
    if command -v curl >/dev/null 2>&1; then
      curl -L "${url}" -o "${archive}"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "${archive}" "${url}"
    else
      echo "Need curl, wget, or gdown to download data." >&2
      exit 1
    fi

    if file "${archive}" | grep -qi 'zip'; then
      unzip -q "${archive}" -d "${tmp_dir}"
    else
      tar -xf "${archive}" -C "${tmp_dir}"
    fi
  fi

  local source_root
  source_root="$(find "${tmp_dir}" -mindepth 1 -maxdepth 2 -type d \( -name baby -o -name tiktok -o -name allrecipes \) -printf '%h\n' | head -1)"
  if [[ -z "${source_root}" ]]; then
    echo "Could not find dataset folders in downloaded data." >&2
    exit 1
  fi

  install_from_local_source "${source_root}"
}

mkdir -p "${DATA_DIR}"

if [[ -n "${MODICF_DATA_URL}" ]]; then
  install_from_archive_url "${MODICF_DATA_URL}"
elif [[ -d "${LOCAL_SOURCE}" ]]; then
  install_from_local_source "${LOCAL_SOURCE}"
else
  cat >&2 <<EOF
No local data source found at:
  ${LOCAL_SOURCE}

Either set LOCAL_SOURCE to a directory containing baby/tiktok/allrecipes,
or set MODICF_DATA_URL to a downloadable archive or Google Drive folder URL.

Example:
  LOCAL_SOURCE=/path/to/modicf_author_data_download bash scripts/download_author_data.sh
  MODICF_DATA_URL='https://drive.google.com/drive/folders/1Vb8iuk3J8-mcBxicFQt_bStZmU1zdE6R?usp=drive_link' bash scripts/download_author_data.sh
EOF
  exit 1
fi
