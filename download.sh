mkdir data
cd data
mkdir intent
mkdir slot
cd intent
wget https://www.dropbox.com/s/k6knxzg30xlk9nv/train.json?dl=1 -O train.json
wget https://www.dropbox.com/s/mkd5on6bwgxkk8t/test.json?dl=1 -O test.json
wget https://www.dropbox.com/s/aknf79lojneid1g/eval.json?dl=1 -O eval.json
cd ..
cd slot
wget https://www.dropbox.com/s/cccw33ap7c7ggcf/train.json?dl=1 -O train.json
wget https://www.dropbox.com/s/wy885sj6k8el65i/test.json?dl=1 -O test.json
wget https://www.dropbox.com/s/x5i9wx59yyj56qg/eval.json?dl=1 -O eval.json
cd ..
cd ..
mkdir cache
cd cache
mkdir intent
mkdir slot
cd intent
wget https://www.dropbox.com/s/e49enf79xd99u1q/intent2idx.json?dl=1 -O intent2idx.json
cd ..
cd slot
wget https://www.dropbox.com/s/65b9903a1r08zla/tag2idx.json?dl=1 -O tag2idx.json
cd ..
cd ..
wget https://www.dropbox.com/s/fk3q742f73r8jtl/dictionary.json?dl=1 -O dictionary.json

