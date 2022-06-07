> Code for ACL 2022 paper 'Towards Making the Most of Cross-Lingual Transfer for Zero-Shot Neural Machine Translation' 
> 
> [[arxiv](https://arxiv.org/abs/2110.08547)] [[ACL anthology](https://aclanthology.org/2022.acl-long.12/)]

----

The code is based on [fairseq v0.10.0](https://github.com/pytorch/fairseq/tree/v0.10.0). The official repo provides more details about the instruction and examples about fairseq toolkit. Three steps are explained as follows to replicate the experiments in the paper.

```bash
## create and activate your conda env first
cd /path/to/this/repo
python -m pip install -e . --user   ## install fairseq toolkit with this repo
python setup.py build_ext --inplace
```


## Step 1: preprocess and binarize data

First download the training/validation/test data from WMT/WAT/CC-align/FLores/Tatoba etc., the detailed urls are in the paper appendix. Suppose all files are placed under a directionary `raw=/path/to/your/raw/data`. Then use the fairseq preprocess.py to binary the dataset: `bash scripts/preprocess.sh`, more details are in the shell scripts.


## Step 2: Train the SixT+ model with two-stage training

The proposed SixT+ model can be trained with the processed dataset. `bash scripts/run_train_sixtp.sh` to train SixT+ model in two training stages. More details are in the shell scripts.



## Step 3: Test the SixT+ model in the testsets

After the model is trained, you can directly test it in a zero-shot manner. The testset are also needed to be binaried with the `preprocess.sh` script. See more in `scripts/run_train_sixtp.sh`. 



## Model Checkpoints 

The many-to-English SixT+ model checkpoint [[download (5.6GB)](https://publicmodel.blob.core.windows.net/sixt/x2e.pt)]

The many-to-many SixT+ model checkpoint [[download (5.6GB)](https://publicmodel.blob.core.windows.net/sixt/x2x.pt)]


## Citation

```
@inproceedings{Chen2022towards,
    title={Towards Making the Most of Multilingual Pretraining for Zero-Shot Neural Machine Translation}, 
    author = "Chen, Guanhua  and
      Ma, Shuming  and
      Chen, Yun  and
      Zhang, Dongdong  and
      Pan, Jia  and
      Wang, Wenping  and
      Wei, Furu",
    year={2022},
    booktitle = "Proceedings of ACL",
    year = "2022",
    page = "142 -- 157",
}
```
