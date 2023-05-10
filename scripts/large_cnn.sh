if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./save_figs" ]; then
    mkdir ./save_figs
fi

if [ ! -d "./save_results" ]; then
    mkdir ./save_results
fi

model='large_cnn'
rounds=100
frac=0.1
local_ep=5
local_bs=10
top_percent=1

python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --iid\
  --local_buffer\
  --global_buffer\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_iid_localB_globalB'.log
  
python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --local_buffer\
  --global_buffer\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_noniid_localB_globalB'.log
  
python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --iid\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_iid'.log
  
python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_noniid'.log

top_percent=0.1

python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --iid\
  --local_buffer\
  --global_buffer\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_iid_localB_globalB'.log 

python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --iid\
  --local_buffer\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_iid_localB'.log 
  
python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --iid\
  --global_buffer\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_iid_globalB'.log 
  
python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --iid\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_iid'.log 
  
python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --local_buffer\
  --global_buffer\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_noniid_localB_globalB'.log 

python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --local_buffer\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_noniid_localB'.log 
  
python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --global_buffer\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_noniid_globalB'.log 
  
python -u main_fedfb.py \
  --model $model\
  --rounds $rounds\
  --frac $frac\
  --local_ep $local_ep\
  --local_bs $local_bs\
  --top_percent $top_percent>logs/$model'_rounds='$rounds'_C='$frac'_E='$local_ep'_B='$local_bs'_alpha='$top_percent'_noniid'.log 