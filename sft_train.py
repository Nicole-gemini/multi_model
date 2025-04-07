# sft_train.py
import torch
from transformers import (Trainer, TrainingArguments, AutoModelForCausalLM, 
                          AutoTokenizer, AutoProcessor,AutoConfig)
from pretrain import MultiModalConfig,MultiModalModel
from dataprocessor import MultiModalDataPreprocessor,UnifiedDataset

class SFTCollator:
    # sft数据处理器
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

def train_sft():
    # 加载配置&模型
    config = MultiModalConfig()
    txt_device = "cuda:1"
    vis_processor = AutoProcessor.from_pretrained(config.vision_model)
    text_tokenizer = AutoTokenizer.from_pretrained(config.text_model)
    AutoConfig.register("multimodal_model",MultiModalConfig)
    AutoModelForCausalLM.register(MultiModalConfig, MultiModalModel)
    model = AutoModelForCausalLM.from_pretrained('/home/featurize/data/multimodal/pretrain/model')

    # 数据预处理器
    processor = MultiModalDataPreprocessor(config)
    dataset = UnifiedDataset(processor, 'sft')
    collator = SFTCollator(text_tokenizer)

    
    # # DeepSpeed训练配置
    args = TrainingArguments(
        output_dir="/home/featurize/data/multimodal/sft/model",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        fp16=True,
        deepspeed="/home/featurize/data/multimodal/df.json"
    )
    
    # 启动训练
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
    train_sft()