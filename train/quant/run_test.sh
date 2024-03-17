exp_name=$1
device=$2
CUDA_VISIBLE_DEVICES=${device} \
python test.py \
--split_ctx dn53 \
--dataset /home/porsche/herok97/fctm-1.0.1/fctm/train/quant/images_test \
--fenet_weight /home/porsche/herok97/fctm-1.0.1/fctm/train/original_weight_m65715/fenet_dn53.pth \
--drnet_weight /home/porsche/herok97/fctm-1.0.1/fctm/train/original_weight_m65715/drnet_dn53.pth \
--inner_codec_weight /home/porsche/herok97/fctm-1.0.1/fctm/train/original_weight_m65715/inner_codec_quant_dn53.pth \
--bin_path /home/porsche/herok97/fctm-1.0.1/fctm/train/quant/results \
--exp_name ${exp_name} \
--enc_device cuda \
--dec_device cpu \
--encode True \
--decode True \
--quant True

# --fenet_weight ./m66342_weight_240302/fenet_seg.pth \
# --drnet_weight ./m66342_weight_240302/drnet_seg.pth \
# --inner_codec_weight ./m66342_weight_240302/inner_codec_seg.pth \
