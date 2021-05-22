#!/bin/bash

mkdir saved_runs/$2

echo "Transferring model to local disk at $2..."
scp -r $1:~/FlyBitch/runs/ saved_runs/$2/

echo "Backup model on remote server..."
ssh $1 "mv ~/FlyBitch/runs/ saved_runs/$2/"
