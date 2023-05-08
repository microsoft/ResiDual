# ResiDual Transformer
**The code implementation of [ResiDual: Transformer with Dual Residual Connections](https://arxiv.org/abs/2304.14802).**

## Enviroment setup
Please use the anaconda to setup the environment by `conda env create -f environment.yaml`.
Then you can activate the environment by `conda activate resi_dual`.

## Model training

For model training, please follow the [guide](https://github.com/facebookresearch/fairseq/tree/main/examples/translation) to process data, and use the command below
```bash
LAYER=6 # or 12
DATA_PATH=THE_PATH_TO_YOUR_DATA_BIN # the data should be fairseq data-bin format
CONFIG=iwslt.yaml # or other config under hydra_config folder

fairseq-hydra-train \
common.user_dir=$(pwd)/resi_dual \
task.data=${DATA_PATH} \
model.encoder.layers=${LAYER} \
model.decoder.layers=${LAYER} \
--config-dir hydra_config \
--config-name ${CONFIG}
```



## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Cite
```
@article{xie2023residual,
  title={ResiDual: Transformer with Dual Residual Connections},
  author={Xie, Shufang and Zhang, Huishuai and Guo, Junliang and Tan, Xu and Bian, Jiang and Awadalla, Hany Hassan and Menezes, Arul and Qin, Tao and Yan, Rui},
  journal={arXiv preprint arXiv:2304.14802},
  year={2023}
}
```
