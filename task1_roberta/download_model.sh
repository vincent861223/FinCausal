#!/bin/bash
# https://drive.google.com/file/d/1NM1UDdZnwHwiNDxhcP-nndaWj24m-90L/view
fileId=1RTwP3nVI3rI_UNqztmZ1fpFaTXQPOo-w
fileName=my_model.pth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}
mv my_model.pth Fincausal_task1_roberta.pth
