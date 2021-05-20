#!/bin/bash

mkdir saved_runs/$2
scp -r $1:~/FlyBitch/runs/ saved_runs/$2
