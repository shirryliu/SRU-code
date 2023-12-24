# SRU-code
This is a PyTorch implementation for the paper accepted by WSDM 2024 (oral presentation):
> Xin Xin, Liu Yang, Ziqi Zhao, Pengjie Ren, Zhumin Chen, Jun Ma, Zhaochun Ren. On the Effectiveness of Unlearning in Session-Based Recommendation.

# Abstract
Session-based recommendation predicts users' future interests from previous interactions in a session. Despite the memorizing of historical samples, the request of unlearning, i.e., to remove the effect of certain training samples, also occurs for reasons such as user privacy or model fidelity. However, existing studies on unlearning are not tailored for the session-based recommendation. On the one hand, these approaches cannot achieve satisfying unlearning effects due to the collaborative correlations and sequential connections between the unlearning item and the remaining items in the session. On the other hand, seldom work has conducted the research to verify the unlearning effectiveness in the session-based recommendation scenario.
In this paper, we propose SRU, a session-based recommendation unlearning framework, which enables high unlearning efficiency, accurate recommendation performance, and improved unlearning effectiveness in session-based recommendation. Specifically, we first partition the training sessions into separate sub-models according to the similarity across the sessions, then we utilize an attention-based aggregation layer to fuse the hidden states according to the correlations between the session and the centroid of the data in the sub-model. To improve the unlearning effectiveness, we further propose three extra data deletion strategies, including collaborative extra deletion (CED), neighbor extra deletion (NED), and random extra deletion (RED). Besides, we propose an evaluation metric that measures whether the unlearning sample can be inferred after the data deletion to verify the unlearning effectiveness. We implement SRU with three representative session-based recommendation models and conduct experiments on three benchmark datasets. Experimental results demonstrate the effectiveness of our methods.

# Run
```
python -u main.py --model=sasrec --sub_model=sasrec --shards=8 --agg_lr=1e-3 --partition=similarity --maxlen=10 \
--dataset=beauty --lr=0.001 --num_epochs=100 --epoch_agg=10 --del_num=-1
```

# Cite
The final version of this paper is coming soon.
