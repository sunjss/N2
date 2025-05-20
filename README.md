[NeurIPS 2024] Towards Dynamic Message Passing on Graphs ($N^2$)
---

This is the official implementation of Towards Dynamic Message Passing on Graphs.
<!-- This is the official implementation of [Towards Dynamic Message Passing on Graphs](#). -->

![N2 Highlight](main.png)

# Preparation
## Requirements

- Python 3.7+
- PyTorch 1.10.0+
- PyTorch Geometric 2.1.0+

Our library versions are as follows:

- torch 2.0.0+cu117
- torch-geometric 2.3.0
- torch-scatter 2.1.1+cu117
- torch-sparse 0.6.17+cu117
- torchmetrics 1.2.0

## Datasets

The datasets should be organized as:

```
.
├── data
│   └── ogb
│   └── coauthor
│   └── ...
│       
```

# Getting Started

- Train with command line

```bash
python train.py --cuda_num 'your chosen cuda' --nbatch 1 --testmode 'your output folder/' --dataset 'dataset name'
```

# Citation
If you find this repository useful in your research, please consider citing:)

```
@InProceedings{sun24neurips,
  title = {Towards Dynamic Message Passing on Graphs},
  author = {Sun, Junshu and Yang, Chenxue and Ji, Xiangyang and Huang, Qingming and Wang, Shuhui},
  booktitle = {Thirty-Eighth Annual Conference on Neural Information Processing Systems},
  year = {2024}
}
```
