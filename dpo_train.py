# dpo_train.py
import torch
from transformers import Trainer, TrainingArguments,AutoConfig,AutoModelForCausalLM,AutoTokenizer
from pretrain import MultiModalConfig,MultiModalModel
from dataprocessor import MultiModalDataPreprocessor,UnifiedDataset

class DPOCollator:
    """DPO数据批处理器"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        # 返回填充处理后的数据
        max_len = max(len(x["query"]) for x in batch)
        return {
            "query": self._pad_tensor([x["query"] for x in batch], max_len).to('cuda:1'),
            "positive": self._pad_tensor([x["positive"] for x in batch], max_len).to('cuda:1'),
            "negative": self._pad_tensor([x["negative"] for x in batch], max_len).to('cuda:1'),
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]).to('cuda:0')
        }
    
    def _pad_tensor(self, sequences, max_len):
        # 填充的方法
        return torch.stack([
            torch.cat([t, torch.full((max_len-len(t),), self.tokenizer.pad_token_id)])
            for t in sequences
        ])

class DPOTrainer(Trainer):
    # 带动态门控的DPO训练器
    def compute_loss(self, model, inputs, return_outputs=False):
        # 获取正负样本
        pixel_values = inputs["pixel_values"]
        pos_texts = [q + p for q, p in zip(inputs["query"], inputs["positive"])]
        neg_texts = [q + n for q, n in zip(inputs["query"], inputs["negative"])]
        
        # 正样本前向
        pos_inputs = self._prepare_inputs(pos_texts, pixel_values)
        pos_outputs = model(**pos_inputs)
        
        # 负样本前向
        neg_inputs = self._prepare_inputs(neg_texts, pixel_values)
        neg_outputs = model(**neg_inputs)
        
        # DPO损失计算
        loss = -torch.log(
            torch.sigmoid(pos_outputs.logits.mean() - neg_outputs.logits.mean())
        )
        return (loss, {"loss": loss}) if return_outputs else loss
    
    def _prepare_inputs(self, texts, pixel_values):
        """动态生成模型输入"""
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        ).to(pixel_values.device)
        inputs["pixel_values"] = pixel_values
        return inputs

def train_dpo():
    # 加载配置&模型
    config = MultiModalConfig()
    AutoConfig.register("multimodal_model",MultiModalConfig)
    AutoModelForCausalLM.register(MultiModalConfig, MultiModalModel)
    text_tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    model = AutoModelForCausalLM.from_pretrained("/home/featurize/data/multimodal/sft/model", config=config)
    
    # 数据预处理器
    processor = MultiModalDataPreprocessor(config)
    dataset = UnifiedDataset(processor, 'dpo')
    collator = DPOCollator(text_tokenizer)
    # DPO训练参数
    args = TrainingArguments(
        output_dir="/home/featurize/data/multimodal/dpo/model",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        fp16=True,
        deepspeed="/home/featurize/data/multimodal/df.json"
    )
    

    
    # 启动训练
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()
    
if __name__ == "__main__":
    train_dpo()