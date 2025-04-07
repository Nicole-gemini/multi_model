# test.py
import torch
from transformers import AutoTokenizer,AutoProcessor,AutoModelForCausalLM
from PIL import Image
from dataprocessor import config
from clip import CLIPModel

class MultimodalTester:
    # 多模态模型测试类
    def __init__(self, model_path):
        # 加载模型和处理器
        self.vis_device = "cuda:0"
        self.txt_device = "cuda:1"
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config['text_model_path'])
        self.processor = AutoProcessor.from_pretrained(config['vis_model_path'])
        self.model.to(self.txt_device).eval()
    
    def generate(self, image_path, prompt, max_len=100, temperature = 0.0, top_k = None):
        # 通用生成方法

        # 处理输入
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(
            images=image, 
            return_tensors="pt"
        ).pixel_values.to(self.vis_device)
        
        # 构建文本输入
        text = self.tokenizer.apply_chat_template(
            [{"role":"system", "content":'You are a helpful assistant.'},
             {"role":"user", "content": f'{prompt}\n<image>'}],
            tokenize=False,
            add_generation_prompt=True).replace('<image>', '<|image_pad|>'*49)
        
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.txt_device)
        
        # 生成输出
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_length=max_len,
                do_sample=(temperature > 0 or (top_k is not None and top_k > 0)),
                temperature=temperature if temperature > 0 else 1.0,
                top_k=top_k if top_k is not None else 0
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

def run_tests():
    # 各个模型测试
    test_image = "/home/featurize/data/multimodal/test/demo.jpg"
    test_prompt = "描述图片中的主要内容"
    # test_prompt = "结合上次对话，重新描述我最在意的内容"
    
    # 测试预训练模型
    print("=== Pretrain Model Test ===")
    pretrain_tester = MultimodalTester("/home/featurize/data/multimodal/pretrain/model")
    print(pretrain_tester.generate(test_image, test_prompt))
    # 对齐度评估
    print("\n=== Alignment Evaluation OF Pretrain ===")
    evaluate_alignment(dpo_tester, test_image, test_prompt)
    
    # 测试SFT模型
    print("\n=== SFT Model Test ===")
    sft_tester = MultimodalTester("/home/featurize/data/multimodal/sft/model")
    print(sft_tester.generate(test_image, test_prompt))
    # 对齐度评估
    print("\n=== Alignment Evaluation OF SFT ===")
    evaluate_alignment(dpo_tester, test_image, test_prompt)
    
    # 测试DPO模型
    print("\n=== DPO Model Test ===")
    dpo_tester = MultimodalTester("/home/featurize/data/multimodal/dpo/model")
    print(dpo_tester.generate(test_image, test_prompt))
    # 对齐度评估
    print("\n=== Alignment Evaluation OF DPO ===")
    evaluate_alignment(dpo_tester, test_image, test_prompt)

def evaluate_alignment(tester, image_path, prompt):
    """图文对齐度评估"""
    
    
    clip_model = CLIPModel.from_pretrained("/home/featurize/data/openai/clip-vit-base-patch32")
    
    # 获取生成文本
    generated_text = tester.generate(image_path, prompt)
    
    # 计算CLIP分数
    image = Image.open(image_path)
    inputs = clip_model.processor(
        text=generated_text, 
        images=image, 
        return_tensors="pt"
    )
    outputs = clip_model(**inputs)
    score = outputs.logits_per_image.item()
    
    print(f"CLIP Alignment Score: {score:.4f}")

if __name__ == "__main__":
    run_tests()