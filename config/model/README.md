# 模型配置文件目录

此目录包含 nanomind 模型的架构配置文件。

## 文件清单

| 文件 | 说明 | 状态 |
|------|------|------|
| nanomind_1b_moe.yaml | 1.26B MoE 模型配置 | 🚧 规划中 |

## 配置示例

```yaml
# nanomind_1b_moe.yaml
model:
  model_type: "qwen3_next_moe"
  architectures: ["Qwen3NextMoeForCausalLM"]
  
  # 词表
  vocab_size: 36005
  
  # 基础架构
  hidden_size: 1152
  num_hidden_layers: 20
  
  # Gated Attention
  num_attention_heads: 8
  num_key_value_heads: 2
  head_dim: 256
  attn_output_gate: true
  
  # Gated DeltaNet
  linear_num_key_heads: 16
  linear_num_value_heads: 16
  linear_key_head_dim: 128
  linear_value_head_dim: 128
  
  # 混合布局: 15层 DeltaNet + 5层 Attention
  full_attention_layer_indexes: [3, 7, 11, 15, 19]
  
  # MoE 配置
  num_experts: 32
  num_experts_per_tok: 3
  num_shared_experts: 1
  moe_intermediate_size: 448
  router_aux_loss_coef: 0.001
  
  # 其他
  hidden_act: "silu"
  max_position_embeddings: 4096
  rope_theta: 10000000.0
  partial_rotary_factor: 0.25
  rms_norm_eps: 1.0e-6
  tie_word_embeddings: true
  torch_dtype: "bfloat16"
  
  # 特殊 Token ID
  pad_token_id: 36000
  eos_token_id: 36002
  bos_token_id: null
```
