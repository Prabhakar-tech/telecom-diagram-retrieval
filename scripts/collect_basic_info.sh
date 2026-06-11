#!/usr/bin/env bash

CSV="/DATA1/prabhakar/telecom/All Images Path.csv"
IMG_DIR="/DATA1/prabhakar/telecom/extracted_images/images"
WORK="/DATA5/prabhakar/telecom_retrieval"

echo "=============================="
echo "BASIC SERVER INFO"
echo "=============================="
date
hostname
whoami
pwd

echo
echo "=============================="
echo "DISK SPACE"
echo "=============================="
df -h /DATA1 /DATA5 || true

echo
echo "=============================="
echo "RAM"
echo "=============================="
free -h || true

echo
echo "=============================="
echo "CPU"
echo "=============================="
lscpu | egrep "Model name|CPU\\(s\\)|Thread|Core|Socket|Architecture" || true

echo
echo "=============================="
echo "GPU"
echo "=============================="
nvidia-smi || echo "nvidia-smi not available"

echo
echo "=============================="
echo "PYTHON ENVIRONMENT"
echo "=============================="
which python3 || true
python3 --version || true
python3 -m pip --version || true

echo
echo "Installed important packages:"
python3 -m pip list | grep -iE "torch|torchvision|transformers|sentence|faiss|qdrant|chromadb|pillow|opencv|pandas|numpy|sklearn|scikit|accelerate|bitsandbytes" || true

echo
echo "=============================="
echo "CSV FILE INFO"
echo "=============================="
ls -lh "$CSV" || true
wc -l "$CSV" || true

echo
echo "First 10 lines of CSV:"
head -10 "$CSV" || true

echo
echo "=============================="
echo "IMAGE FOLDER INFO"
echo "=============================="
ls -ld "$IMG_DIR" || true
du -sh "$IMG_DIR" || true

echo
echo "Total files recursively:"
find "$IMG_DIR" -type f | wc -l || true

echo
echo "Image extension counts:"
find "$IMG_DIR" -type f | awk '
{
  n=split($0,a,".");
  if(n>1){
    ext=tolower(a[n]);
    count[ext]++;
  }
}
END{
  for(e in count) print count[e], e
}' | sort -nr | head -30 || true

echo
echo "First 20 image/file paths:"
find "$IMG_DIR" -type f | head -20 || true

echo
echo "=============================="
echo "WORKSPACE INFO"
echo "=============================="
ls -lh "$WORK" || true
