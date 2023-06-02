#!/bin/bash

while getopts ":v:m:" opt; do
  case $opt in
    v) version="$OPTARG";;
    m) model="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2;;
  esac
done

echo "Version: $version"
echo "Model: $model"
