#!/bin/bash

cd /users/yhb18174/Recreating_DMTA/

git checkout stage

git add .

git commit -m "Automatic commit on $(date +\%Y-\%m-\%d \%H:\%M)"

git push -f github stage
