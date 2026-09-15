#!/usr/bin/env bash
set -euo pipefail
exec ssh -o IdentityAgent=/tmp/ssh-XXXXXX0rvlUn/agent.2039712 -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=10 -i /home/aiden/.ssh/id_ed25519_macbook_aiden -p 22 v4@115.145.175.197 "$@"
