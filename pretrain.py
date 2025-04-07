# pretrain.py
import torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoModelForCausalLM, Trainer, 
                          TrainingArguments,PretrainedConfig, 
                          PreTrainedModel,AutoProcessor,AutoTokenizer)
from dataprocessor import UnifiedDataset,config,MultiModalDataPreprocessor
from transformers.modeling_outputs import CausalLMOutputWithPast

class pretrainCollator:
    # 数据处理器
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        # 返回填充处理后的数据
        max_len = max(len(x["input_ids"]) for x in batch)
        return {
            "input_ids": self._pad_tensor([x["input_ids"] for x in batch], max_len).to('cuda:1'),
            "labels": self._pad_tensor([x["labels"] for x in batch], max_len).to('cuda:1'),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]).to('cuda:0')
        }
    
    def _pad_tensor(self, sequences, max_len):
        # 填充的方法
        return torch.stack([
            torch.cat([t, torch.full((max_len-len(t),), self.tokenizer.pad_token_id)])
            for t in sequences
        ])
    
class MultiModalConfig(PretrainedConfig):
    # 模型配置类
    model_type = "multimodal_model"
    def __init__(self,config=config, **kwargs):
        super().__init__(**kwargs)
        self.vision_model = config['vis_model_path']
        self.text_model = config['text_model_path']
        self.image_tokens = config['image_tokens']
        self.mem_length = 3 # 记忆长度
        self.freeze_vision = True # 是否冻结视觉模型
        self.freeze_text = False # 是否冻结文本模型



class MultiModalModel(PreTrainedModel):
    config_class = MultiModalConfig
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        # 设备分配，实现混合并行
        # 加载视觉模型、文本模型与分词器
        self.vis_device = "cuda:0"
        self.txt_device = "cuda:1"
        self.vision_encoder = AutoModel.from_pretrained(config.vision_model).to(self.vis_device)
        self.text_encoder = AutoModelForCausalLM.from_pretrained(config.text_model).to(self.txt_device)      
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.text_model)
        
        # 视觉投影层
        vision_dim = self.vision_encoder.config.vision_config.hidden_size
        text_dim = self.text_encoder.config.hidden_size
        self.vis_linear1 = torch.nn.Linear(vision_dim*4, text_dim).to(self.vis_device)
        self.vis_linear2 = torch.nn.Linear(text_dim, text_dim).to(self.vis_device)

        # 多轮记忆
        self.mem = torch.nn.Parameter(torch.zeros(config.mem_length, text_dim, device=self.txt_device), requires_grad=False)
        
        # 用于动态门控的投影层
        self.fusion_gate = torch.nn.Linear(2*text_dim, text_dim)

        #  冻结模型参数
        if config.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        # 冻结文本模型参数（本次文本模型不冻结）
        if config.freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, pixel_values, labels):
        # 数据路由策略，实现混合并行
        pixel_values = pixel_values.to(self.vis_device)
        input_ids = input_ids.to(self.txt_device)
        labels = labels.to(self.txt_device) if labels is not None else None
        # 图像特征提取
        with torch.cuda.amp.autocast(device_type=self.vis_device.split(':')[0]):
            vis_feat = self.vision_encoder(pixel_values).last_hidden_state
            b,_,d = vis_feat.shape
            vis_feat = vis_feat.view(b, -1, d*4) # 展平图像特征
            vis_feat = self.vis_linear1(vis_feat)
            vis_feat = self.vis_linear2(F.silu(vis_feat))


        # 文本词嵌入
        with torch.cuda.amp.autocast(device_type=self.txt_device.split(':')[0]):
            txt_emb = self.text_encoder.get_input_embeddings()(input_ids)

            # 保持类型一致
            vis_feat = vis_feat.to(self.txt_device, txt_emb.dtype)
        
            # 三轮记忆处理
            mem_mean = self.mem.mean(dim=0, keepdim=True)
            mem_broad = mem_mean.expand(b, vis_feat.size(1), -1)

            # 动态门控融合
            gate_score = self.fusion_gate(torch.cat([vis_feat, mem_broad], dim=-1))
            gate_score = torch.sigmoid(gate_score)
            fused = gate_score * vis_feat + (1 - gate_score) * mem_broad

            # 更新记忆
            tmp = fused.mean(dim=(0,1),keepdim=True).detach()
            self.mem.data[:-1] = self.mem.data[1:]
            self.mem.data[-1] = tmp

            # 用动态融合的结果替换图像占位符 （patch级插入）
            merged_emb = self._merge_gate_mem(txt_emb, input_ids, fused)

            # 因果语言建模
            res = self.text_encoder(inputs_embeds=merged_emb, labels=labels)

            # 返回损失
            logits = res[0]
            cross_loss = None
            if labels is not None:
                labels = labels.to(logits.device)
                # 计算交叉熵损失
                cross_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1), 
                    ignore_index=self.text_tokenizer.pad_token_id)
            
        return CausalLMOutputWithPast(
                loss=cross_loss,
                logits=logits,
                past_key_values=res.past_key_values
        )

    def _merge_gate_mem(self, txt_emb, input_ids, fused):
        final_emb = txt_emb.clone()
        # 获取文本中图像占位符的位置
        img_pad_id = self.text_tokenizer('<image_pad>')['input_ids'][0]
        b,s,h = txt_emb.shape # 批次大小，序列长度，隐藏维度
        _, pad_num, _ = fused.shape # 批次大小，图像占位符数量，隐藏维度

        row_id = []
        col_id = []
        patch_id = []

        for b_id in range(b):
            img_pos = (input_ids[b] == img_pad_id).nonzero(as_tuple=True)[0]

            if img_pos.numel() != pad_num:
                raise ValueError(f"图像占位符数量不匹配: {img_pos.numel()} != {pad_num}")
            # 保证顺序
            img_pos,_ = img_pos.sort()
            # 获取对应批次的fused的patch, 得到(pad_num, hidden_dim)
            fused_b = fused[b_id]
            # 获取对应的位置信息与融合信息
            row_id.append(torch.full((pad_num,), b_id))
            col_id.append(img_pos)
            patch_id.append(fused_b)
        
        # 处理遍历后的位置信息
        row_id_tensor = torch.cat(row_id, dim=0)
        col_id_tensor = torch.cat(col_id, dim=0)
        patch_id_tensor = torch.cat(patch_id, dim=0)

        # 替换
        final_emb[row_id_tensor, col_id_tensor, :] = patch_id_tensor

        return final_emb
    
    def reset_memory(self):
        # 清空记忆
        self.mem.data.zero_()
            
def train():
    config = MultiModalConfig()
    model = MultiModalModel(config).cuda()
    
    # 数据预处理器
    processor = MultiModalDataPreprocessor(config)
    dataset = UnifiedDataset(processor, data_type="pretrain")
    collator = pretrainCollator(model.text_tokenizer)
    
    # DeepSpeed训练配置
    args = TrainingArguments(
        output_dir="/home/featurize/data/multimodal/pretrain/model",
        do_train=True,
        deepspeed="/home/featurize/data/multimodal/df.json",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        fp16=True,
        optim="adamw_torch",
        report_to="tensorboard",
        ddp_find_unused_parameters=False
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()

if __name__ == "__main__":
    train()