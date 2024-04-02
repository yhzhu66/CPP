# Chain of Propagation Prompting for Node Classification

This is the implementation of our paper 'Chain of Propagation Prompting for Node Classification' 

In this paper, we propose using a simple pattern of message-passing, which is based on a weighted multi-hop graph, to prompt self-attention to capture a more complex message-passing pattern, 
resulting in the maximal receptive field and reduced reliance on label information. Also, a majority voting-like module is used to improve predictive confidence.

The paper can be viewed via https://dl.acm.org/doi/10.1145/3581783.3612431, and the framework of our CPP is listed in the following.
![image](https://github.com/yhzhu66/CPP/assets/52006047/656f42ff-5eba-445c-9173-f0af20686f82)



If you are interested in our work, please also cite our paper:

@inproceedings{zhu2023chain,
  title={Chain of Propagation Prompting for Node Classification},
  author={Zhu, Yonghua and Deng, Zhenyun and Chen, Yang and Amor, Robert and Witbrock, Michael},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={3012--3020},
  year={2023}
}
