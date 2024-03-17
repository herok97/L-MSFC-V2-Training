exp_name=$1

CUDA_VISIBLE_DEVICES=4 \
python test.py \
--split_ctx dn53 \
--dataset /home/porsche/herok97/fctm-1.0.1/fctm/train/quant/images_test \
--fenet_weight /home/porsche/herok97/fctm-1.0.1/fctm/train/split/checkponts/tvd_cttc/fenet_dn53.pth \
--drnet_weight /home/porsche/herok97/fctm-1.0.1/fctm/train/split/checkponts/tvd_cttc/drnet_dn53.pth \
--inner_codec_weight /home/porsche/herok97/fctm-1.0.1/fctm/train/split/checkponts/tvd_cttc/inner_codec_quant_dn53.pth \
--bin_path /home/porsche/herok97/fctm-1.0.1/fctm/train/quant/results \
--exp_name ${exp_name} \
--enc_device cuda \
--dec_device cuda \
--encode True \
--decode True \
--quant True