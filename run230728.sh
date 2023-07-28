# SEG
python main.py --config SEG/semi/10/ODOC_semi10
python main.py --config SEG/semi/10/ODOC_semi20
python main.py --config SEG/semi/10/ODOC_semi30
python main.py --config SEG/semi/10/ODOC_semi40
python main.py --config SEG/semi/10/ODOC_semi50
python main.py --config SEG/semi/10/ODOC_semi60
python main.py --config SEG/semi/10/ODOC_semi70
python main.py --config SEG/semi/10/ODOC_semi80
python main.py --config SEG/semi/10/ODOC_semi90


# 先创建顶级目录temp
mkdir temp

# 添加temp的写权限
chmod 777 temp

# 在temp下创建子目录
mkdir temp/random1_ODOC_sup10
mkdir temp/random1_ODOC_sup20
mkdir temp/random1_ODOC_sup30
mkdir temp/random1_ODOC_sup40
mkdir temp/random1_ODOC_sup50
mkdir temp/random1_ODOC_sup60
mkdir temp/random1_ODOC_sup70
mkdir temp/random1_ODOC_sup80
mkdir temp/random1_ODOC_sup90

# 在每个子目录下创建ckpt目录
mkdir temp/random1_ODOC_sup10/ckpt
mkdir temp/random1_ODOC_sup20/ckpt
mkdir temp/random1_ODOC_sup30/ckpt
mkdir temp/random1_ODOC_sup40/ckpt
mkdir temp/random1_ODOC_sup50/ckpt
mkdir temp/random1_ODOC_sup60/ckpt
mkdir temp/random1_ODOC_sup70/ckpt
mkdir temp/random1_ODOC_sup80/ckpt
mkdir temp/random1_ODOC_sup90/ckpt

cp sup/random1_ODOC_sup10/ckpt/val_mIoU* /root/autodl-fs/temp/random1_ODOC_sup10/ckpt
cp sup/random1_ODOC_sup20/ckpt/val_mIoU* /root/autodl-fs/temp/random1_ODOC_sup20/ckpt
cp sup/random1_ODOC_sup30/ckpt/val_mIoU* /root/autodl-fs/temp/random1_ODOC_sup30/ckpt
cp sup/random1_ODOC_sup40/ckpt/val_mIoU* /root/autodl-fs/temp/random1_ODOC_sup40/ckpt
cp sup/random1_ODOC_sup50/ckpt/val_mIoU* /root/autodl-fs/temp/random1_ODOC_sup50/ckpt
cp sup/random1_ODOC_sup60/ckpt/val_mIoU* /root/autodl-fs/temp/random1_ODOC_sup60/ckpt
cp sup/random1_ODOC_sup70/ckpt/val_mIoU* /root/autodl-fs/temp/random1_ODOC_sup70/ckpt
cp sup/random1_ODOC_sup80/ckpt/val_mIoU* /root/autodl-fs/temp/random1_ODOC_sup80/ckpt
cp sup/random1_ODOC_sup90/ckpt/val_mIoU* /root/autodl-fs/temp/random1_ODOC_sup90/ckpt



