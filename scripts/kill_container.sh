#!/usr/bin/env bash
set -euo pipefail

removed=false
for CONTAINER in practice4-sim practice4-sim-cpu; do
  if docker inspect "$CONTAINER" &>/dev/null; then
    docker rm -f "$CONTAINER"
    echo "Контейнер '$CONTAINER' удалён."
    removed=true
  fi
done

if ! $removed; then
  echo "Контейнеры practice4-sim / practice4-sim-cpu не найдены."
fi
