#!/usr/bin/env bash
# Worker entrypoint: start sshd when a public key is supplied, then exec the
# worker.
#
# A rented worker exposes only the node-execution bridge. Diagnosing one
# therefore means inferring its state from whether a whole model load passes or
# fails, which costs minutes per hypothesis. With PUBLIC_KEY set the pod also
# answers on 22/tcp and the state can be read directly.
#
# No key means no sshd and no open port — the default is unchanged.
set -euo pipefail

if [ -n "${PUBLIC_KEY:-}" ]; then
  if command -v sshd >/dev/null 2>&1; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    # Host keys are generated per container: the image must not ship a fixed
    # private key, and every pod is a fresh host anyway.
    ssh-keygen -A >/dev/null
    mkdir -p /run/sshd
    # Key-only, root login permitted — the pod has a single account and the
    # key is the sole credential.
    # An sshd session starts from a clean environment, so the container's ENV
    # (PATH into the conda env, HF_HOME on the persistent volume) would be
    # absent and `python` would resolve to the system interpreter. Export the
    # variables that make an interactive session match the worker process.
    {
      echo "export PATH=${PATH}"
      echo "export VIRTUAL_ENV=${VIRTUAL_ENV:-/opt/conda}"
      [ -n "${HF_HOME:-}" ] && echo "export HF_HOME=${HF_HOME}"
    } > /etc/profile.d/nodetool.sh
    chmod 644 /etc/profile.d/nodetool.sh

    /usr/sbin/sshd \
      -o PermitRootLogin=prohibit-password \
      -o PasswordAuthentication=no \
      -o ChallengeResponseAuthentication=no
    echo "sshd listening on 22 (key-only)"
  else
    echo "PUBLIC_KEY is set but sshd is not installed in this image" >&2
  fi
fi

exec "$@"
