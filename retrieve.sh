#!/bin/bash

set -e

if [ -d saved_runs/$2 ]; then
    echo $2 already exists, press enter to continue, or Ctrl+C to cancel.
    read -p "Press [ENTER] "
fi

mkdir saved_runs/$2 || true

echo "Testing connection..."
ssh $1 "echo 'Connection OK'"

echo "Transferring model to local disk at $2..."
rsync -avH $1:~/FlyBitch/runs/ saved_runs/$2/ --compress

echo "Backup model on remote server..."
ssh $1 "rsync -avH ~/FlyBitch/runs/ ~/FlyBitch/saved_runs/$2/"
