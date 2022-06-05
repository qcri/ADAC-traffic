## Run sample ADAC training specifying the environment and buffer

ENV='gharaffa-NTFT20'
BUF_TYPE='stationary' ## 'stationary' for 1-day batch and 'stationary2' for 1-week batch, 'moving' for partialRL both sizes
BUF=$BUF_TYPE'-NTFT20'
SEED=20
ID_TOKEN='dac-r0.5-sample-2-nmode0.8-nostate'

python run-offline-rl.py \
        --env=$ENV \
        --seed=$SEED \
        --buffer_name=$BUF \
        --max_timesteps=100000 \
        --offline_algo=BCQ \
        --mm_threshold=1 \
        --sm_threshold=5 \
        --BCQ_threshold=0.3 \
        --id_tokens=~$ID_TOKEN \
        --dac_configs=1 \
        > dac-bcq-$BUF_TYPE-$ID_TOKEN.log 2>&1 &

