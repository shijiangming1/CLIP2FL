# **CLIP-guided Federated Learning on Heterogeneous and Long-Tailed Data**

This is an official implementation of the following paper:

> Jiangming Shi, Shanshan Zheng, Xiangbo Yin, Yang Lu, Yuan Xie, Yanyun Qu.
>
> **CLIP-guided Federated Learning on Heterogeneous and Long-Tailed Data**
>
> *AAAI Conference on Artificial Intelligence (AAAI), 2024* 


**Abstract:** Federated learning (FL) provides a decentralized machine learning paradigm where a server collaborates with a group of clients to learn a global model without accessing the clients' data. User heterogeneity is a significant challenge for FL, which together with the class-distribution imbalance further enhances the difficulty of FL. Great progress has been made in large vision-language models, such as Contrastive Language-Image Pre-training (CLIP), which paves a new way for image classification and object recognition. Inspired by the success of CLIP on few-shot and zero-shot learning, we use CLIP to optimize the federated learning between server and client models under its vision-language supervision. It is promising to mitigate the user heterogeneity and class-distribution balance due to the powerful cross-modality representation and rich open-vocabulary prior knowledge of CLIP. In this paper, we propose the CLIP-guided FL (CLIP2FL) method on heterogeneous and long-tailed data. In CLIP2FL, the knowledge of the off-the-shelf CLIP model is transferred to the client-server models, and a bridge is built between the client and server. Specifically, for client-side learning, knowledge distillation is conducted between client models and CLIP to improve the ability of client-side feature representation. For server-side learning, in order to mitigate the heterogeneity and class-distribution imbalance, we generate federated features to retrain the server model. A prototype contrastive learning with the supervision of the text encoder of CLIP is introduced to generate federated features depending on the client-side gradients, and they are used to retrain a balanced server classifier. Extensive experimental results on several benchmarks demonstrate that our method achieves impressive performance and effectively deals with data heterogeneity.



### Dependencies

- python 3.7.9 (Anaconda)
- PyTorch 1.7.0
- torchvision 0.8.1
- CUDA 11.2
- cuDNN 8.0.4
- clip-by-openai 0.1



### Dataset

- CIFAR-10
- CIFAR-100
- ImageNet-LT



### Parameters

The following arguments to the `./options.py` file control the important parameters of the experiment.

| Argument                    | Description                                        |
| --------------------------- | -------------------------------------------------  |
| `dataset`                   | Name of dataset                                    |
| `alpha`                     | Controls the distillation weight of the CLIP model.|
| `contrast_alpha`            | Controls the balance of PCL and GML.               |
| `num_classes`               | Number of classes                                  |
| `num_clients`               | Number of all clients.                             |
| `num_online_clients`        | Number of participating local clients.             |
| `num_rounds`                | Number of communication rounds.                    |
| `num_epochs_local_training` | Number of local epochs.                            |
| `batch_size_local_training` | Batch size of local training.                      |
| `match_epoch`               | Number of optimizing federated features.           |
| `crt_epoch`                 | Number of re-training classifier.                  |
| `ipc`                       | Number of federated features per class.            |
| `lr_local_training`         | Learning rate of client updating.                  |
| `lr_feature`                | Learning rate of federated features optimization.  |
| `lr_net`                    | Learning rate of classifier re-training            |
| `non_iid_alpha`             | Control the degree of heterogeneity.               |
| `imb_factor`                | Control the degree of imbalance.                   |



### Usage

Here is an example to run CReFF on CIFAR-10 with imb_factor=0.01:

```python main.py 
--dataset cifar10 \
--num_classes=10 \
--num_rounds=200 \
--match_epoch=100 \
--contrast_alpha=0.001 \
--imb_factor=0.01

```

### Contact
jiangming.shi@outlook.com; S_yinxb@163.com.

The code is implemented based on [CReFF](https://github.com/shangxinyi/CReFF-FL).



