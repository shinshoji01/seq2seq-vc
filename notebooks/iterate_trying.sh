max_retries=200
attempts=0

while [ $attempts -lt $max_retries ]; do
	# HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mntcephfs/lab_data/shoinoue/Models/hf_hub/ python3 xtuner/tools/train.py config/llava_llama3_8b_instruct_quant_clip_vit_large_p14_336_e1_gpu1_pretrain_copy.py && break
     HF_ENDPOINT=https://hf-mirror.com HF_HOME=/mntcephfs/lab_data/shoinoue/Models/hf_hub/ python3 A_download_GLOBE.py && break
	attempts=$((attempts+1))
	echo "Attempt $attempts failed, retrying..."
	sleep 10
done

if [ $attempts -eq $max_retries ]; then
	echo "Command failed after $max_retries attempts."
fi
