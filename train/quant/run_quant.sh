SPLIT_CTX=$1
WEIGHT_DIR=$2

CUDA_VISIBLE_DEVICES=4 \
python quantize.py \
--split_ctx ${SPLIT_CTX} \
--dataset /home/porsche/herok97/fctm-1.0.1/fctm/train/quant/images_calibration \
--fenet_weight ${WEIGHT_DIR}/fenet_${SPLIT_CTX}.pth \
--inner_codec_weight ${WEIGHT_DIR}/inner_codec_${SPLIT_CTX}.pth \
--inner_codec_quant_weight ${WEIGHT_DIR}/inner_codec_quant_${SPLIT_CTX}.pth \
--device cuda \