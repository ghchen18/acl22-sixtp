#!/bin/bash
train_sixtp_first_stage(){
    config='m2En'  ## ['m2En', 'm2m'], train a many-to-English or many-to-many SixT+ model
    task='encfix_embfix_decrand'
    fseq=$WORKLOC/fseq
    modeldir=$WORKLOC/models/sixtp_1stage_${task}_xlmrL_${config} && mkdir -p $modeldir 
    
    if [[ $config == *"m2En"* ]] ; then
        echo "Now train the many-to-English SixT+ model."
        lang_pairs="de-en,es-en,fi-en,hi-en,ru-en,zh-en"
        lang_proj=""
    else
        echo "Now train the many-to-many SixT+ model."
        lang_pairs="de-en,es-en,fi-en,hi-en,ru-en,zh-en,en-de,en-es,en-fi,en-hi,en-ru,en-zh"
        lang_proj="--enable-lang-proj"
    fi 
    lang_dict="en,de,es,fi,hi,ru,zh"
    MaxUpdates=100000

    ## we simulate the 128 GPUs by setting update_freq=8 while using 32 GPUs

    python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT \
        train.py $fseq  --save-dir $modeldir  --seed 16 --fp16  ${lang_proj}  \
        --arch transformer --task translation_multi_simple_epoch --sampling-method 'temperature'  --sampling-temperature 5  \
        --decoder-ffn-embed-dim 3072 --decoder-attention-heads 16 --decoder-layers 12 --encoder-embed-dim 1024 --decoder-embed-dim 1024 \
        --langs ${lang_dict} --lang-pairs ${lang_pairs} --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1  --optimizer adam --adam-betas '(0.9, 0.98)'  --lr-scheduler inverse_sqrt  --lr 5e-04  \
        --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000  --max-update ${MaxUpdates} --dropout 0.1 --attention-dropout 0.0 \
        --weight-decay 0.0  --max-tokens 2048 --update-freq 8 --skip-invalid-size-inputs-valid-test  --log-interval 100 --truncate-source \
        --encoder-normalize-before --decoder-normalize-before --xlmr-task $task --tensorboard-logdir $modeldir/tensorboard \
        --share-all-embeddings --max-source-positions 512  --activation-fn gelu_accurate --xlmr-modeldir $WORKLOC/models/xlmrL_base \
        --log-format 'tqdm' --clip-norm 2.0 --num-workers 0 --save-interval-updates 3000 --mplm-type 'xlmrL' --ddp-backend=legacy_ddp \
    2>&1 | tee $modeldir/train.log  
}

train_sixtp_second_stage(){   
    config='m2En'  ## ['m2En', 'm2m']
    task='xlmr_2stage_posdrop'
    fseq=$WORKLOC/fseq
    modeldir=$WORKLOC/models/sixtp_2stage_${task}_xlmrL_${config} && mkdir -p $modeldir 
    
    if [[ $config == *"m2En"* ]] ; then
        echo "Now train the many-to-English SixT+ model."
        lang_pairs="de-en,es-en,fi-en,hi-en,ru-en,zh-en"
        lang_proj=""
    else
        echo "Now train the many-to-many SixT+ model."
        lang_pairs="de-en,es-en,fi-en,hi-en,ru-en,zh-en,en-de,en-es,en-fi,en-hi,en-ru,en-zh"
        lang_proj="--enable-lang-proj"
    fi 
    lang_dict="en,de,es,fi,hi,ru,zh"
    MaxUpdates=10000

    ## we simulate the 128 GPUs by setting update_freq=16 while using 32 GPUs

    python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT \
        train.py $fseq --save-dir $modeldir  --seed 16 --fp16 --resdrop-layer 22  ${lang_proj} \
        --arch transformer --task translation_multi_simple_epoch --sampling-method 'temperature'  --sampling-temperature 5  \
        --decoder-ffn-embed-dim 3072 --decoder-attention-heads 16 --decoder-layers 12 --encoder-embed-dim 1024 --decoder-embed-dim 1024 \
        --langs ${lang_dict} --lang-pairs ${lang_pairs} --criterion label_smoothed_cross_entropy --same-lang-per-batch --enable-lang-ids \
        --label-smoothing 0.1  --optimizer adam --adam-betas '(0.9, 0.98)'  --lr-scheduler inverse_sqrt  --lr 1e-04 --truncate-source \
        --warmup-updates 10 --warmup-init-lr 0.00002  --max-update ${MaxUpdates} --dropout 0.1 --attention-dropout 0.0 \
        --weight-decay 0.0  --max-tokens 1024 --update-freq 16 --skip-invalid-size-inputs-valid-test  --log-interval 10 \
        --encoder-normalize-before --decoder-normalize-before --xlmr-task $task --tensorboard-logdir $modeldir/tensorboard \
        --share-all-embeddings --max-source-positions 512  --activation-fn gelu_accurate --xlmr-modeldir $WORKLOC/models/xlmrL_base \
        --log-format 'tqdm' --clip-norm 2.0 --num-workers 0 --save-interval-updates 500 --mplm-type 'xlmrL' --no-epoch-checkpoints  \
        --ft-last-f-dir $WORKLOC/models/sixtp_1stage_encfix_embfix_decrand_xlmrL_m2En  --ddp-backend=legacy_ddp \
    2>&1 | tee $modeldir/train.log


}



run_on_testsets(){ 
    task='xlmr_2stage_posdrop'
    src=$1  && tgt=$2
    max_token=5000

    lang_dict="en,de,es,fi,hi,ru,zh"
    config='m2m'  ## ['m2En', 'm2m']
    if [[ $config == *"m2En"* ]] ; then
        echo "Now train the many-to-English SixT+ model."
        lang_pairs="de-en,es-en,fi-en,hi-en,ru-en,zh-en"
        lang_proj=""
        enable_lang_proj='False'
    else
        echo "Now train the many-to-many SixT+ model."
        lang_pairs="de-en,es-en,fi-en,hi-en,ru-en,zh-en,en-de,en-es,en-fi,en-hi,en-ru,en-zh"
        lang_proj="--enable-lang-proj"
        enable_lang_proj='True'
    fi 

    modeldir=$WORKLOC/models/sixtp_2stage_${task}_xlmrL_${config}
    resdir=$modeldir/genres/${src}2${tgt} && mkdir -p $resdir

    fseq=$WORKLOC/fseq  && mkdir -p $fseq
    raw_reference=$WORKLOC/raw/test.${src}-${tgt}.${tgt}

    python generate.py $fseq -s $src -t $tgt  ${lang_proj} \
        --path $modeldir/checkpoint_best.pt --fp16 --same-lang-per-batch --enable-lang-ids \
        --max-tokens ${max_token} --beam 5 --sacrebleu \
        --remove-bpe  --decoding-path $resdir --mplm-type ${MPLM}  --xlmr-task $task \
        --model-overrides "{'xlmr_modeldir':\"${WORKLOC}/models/xlmrL_base\", 'enable_lang_proj':\"${enable_lang_proj}\"}" \
        --langs ${lang_dict} --task translation_multi_simple_epoch 2>&1 | tee $resdir/gen_out

    cat $resdir/gen_out | grep -P "^H" | sort -V | cut -f 3-  > $resdir/decoding.txt
    python scripts/his/spm_decode.py --model $WORKLOC/models/xlmrL_base/sentencepiece.bpe.model \
        --input $resdir/decoding.txt > $resdir/decoding.detok

    echo "====BLEU score for ${src}-2-$tgt is ...."
    cat $resdir/decoding.detok | sacrebleu -b $raw_reference

}

export MPLM=xlmrL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORKLOC=/path/to/your/work/location


##=======================================
# Before training, download the official XLM-R checkpoint and placed in $WORKLOC/models/xlmrL_base/{model.pt,dict.txt,...}
# download and binarize the train/valid/test dataset as in `scripts/preprocess.sh`, and placed in $WORKLOC/fseq 
##=======================================


echo "Step 1: Train the sixtp at the first stage"
train_sixtp_first_stage


echo "Step 2: Train the sixtp at the second stage"
train_sixtp_second_stage


echo "Step 3: test the trained model on NMT testsets"
tgt=en
for src in 'de' 'es' 'ro'  'fi' 'lv' 'et' 'hi' 'ne' 'si' 'gu' 'zh' 'ja' 'ko' 'nl' 'it' 'ru' 'pl' 'tr' 'kk' 'my' 'km' 'ar' 'ps'; do 
    echo "Start to translate ${src} testsets to ${tgt}."
    run_on_testsets  $src $tgt
done
