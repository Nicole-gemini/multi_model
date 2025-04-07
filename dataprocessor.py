# 多模态模型数据预处理脚本（包括预训练、SFT和DPO数据处理）
import os
import json
import random
import jieba
from PIL import Image
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoProcessor
from torch.utils.data import Dataset

config = {
    'img_path': '/home/featurize/data/multimodal/pretrain/images',
    'pretrain_data': '/home/featurize/data/multimodal/pretrain/chat-translated.json',
    'sft_img_path': '/home/featurize/data/multimodal/sft/images',
    'sft_data': '/home/featurize/data/multimodal/sft/llava_instruct_150k.json',
    'vis_model_path': '/home/featurize/data/multimodal/visual_model/siglip-base-patch16-224',
    'text_model_path': '/home/featurize/data/multimodal/text_model/Qwen2.5-1.5B-Instruct',
    'img_pad_token': '<|image_pad|>',
    'img_pad_num': 49
    }

class MultiModalDataPreprocessor:
    # 用于多模态数据预处理的类
    def __init__(self, config=config):
        """
        参数:
            config: dict
                - img_path: 预训练图片目录路径
                - pretrain_data: 预训练数据JSON路径
                - sft_img_path: SFT图片目录路径
                - sft_data: SFT数据JSON路径 
                - vis_model_path: 视觉模型路径
                - text_model_path: 文本模型路径
                - img_pad_token: 图片占位符标记
                - img_pad_num: 图片token数量
        """
        self.config = config
        self.visual_processor = AutoProcessor.from_pretrained(config['vis_model_path'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(config['text_model_path'])
        self.text_tokenizer.add_special_tokens({'additional_special_tokens': config['img_pad_token']})

    # =================================================== 预训练数据处理 ===================================================
    def _load_pretrain_samples(self) -> List[Dict]:
        # 加载原始预训练数据
        with open(self.config['pretrain_data'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _process_pretrain_sample(self, sample: Dict) -> Dict:
        # 处理单个预训练数据样本
        try:
            image = Image.open(os.path.join(self.config['img_path'], sample['image'])).convert('RGB')
            pixel_values = self.visual_processor(images=image, return_tensors='pt')['pixel_values']
            
            # 构建对话模板
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": sample['conversations'][0]['value']},
           ]
            text = self.text_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            ).replace('<image>', self.config['img_pad_token'] * self.config['img_pad_num'])
            ans = sample['conversations'][1]['value']+self.text_tokenizer.eos_token
            # 标记化处理
            text_ids = self.text_tokenizer(text, return_tensors='pt')['input_ids']
            ans_ids = self.text_tokenizer(ans, return_tensors='pt')['input_ids']
            total_ids = text_ids+ans_ids
            labels = [self.text_tokenizer.pad_token_id]*len(text_ids)+ans_ids
            return {
                'input_ids': total_ids[:-1],
                'labels': labels[1:],
                'pixel_values': pixel_values
            }
        except Exception as e:
            return self._create_empty_sample()

    # ================================================== SFT数据处理 ==================================================
    def _load_sft_samples(self) -> List[Dict]:
        # 加载SFT数据
        with open(self.config['sft_data'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _find_response_spans(self, input_ids: List[int]) -> List[Tuple[int, int]]:
        # 定位assistant回复的token位置
        spans = []
        start_token = self.text_tokenizer("assistant")['input_ids'][0]
        end_token = self.text_tokenizer("<|im_end|>")['input_ids'][0]
        
        start_idx = None
        for i, token_id in enumerate(input_ids):
            if token_id == start_token:
                # 遇到assistant起始token，记录位置（这里跳过assistant）
                start_idx = i + 1
            elif token_id == end_token and start_idx is not None:
                # 将起始值和结束值都存储到span，由于后面数据处理需要，这里的结束值+1
                spans.append((start_idx, i+1))
                start_idx = None
        return spans

    def _process_sft_sample(self, sample: Dict) -> Dict:
        # 处理单个SFT数据样本
        try:
            image = Image.open(os.path.join(self.config['sft_img_path'], 'COCO_train2014_'+str(sample['image']))).convert('RGB')
            pixel_values = self.visual_processor(images=image, return_tensors='pt')['pixel_values']
            
            # 多轮对话处理
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for conv in sample['conversations']:
                role = "user" if conv['from'] == 'human' else "assistant"
                messages.append({"role": role, "content": conv['value']})
            
            text = self.text_tokenizer.apply_chat_template(
                messages,
                tokenize=False
            ).replace('<image>', self.config['img_pad_token'] * self.config['img_pad_num'])
            
            # 标记化并设置labels
            text_ids = self.text_tokenizer(text, return_tensors='pt')['input_ids']
            labels = [self.text_tokenizer.pad_token_id]*len(text_ids)


            spans = self._find_response_spans(text_ids.tolist())
            

            for start, end in spans:
                labels[start:end] = text_ids[start:end]
                
            return {
                'input_ids': text_ids[:-1],  # 去掉EOS
                'labels': labels[1:],         # 去掉开头
                'pixel_values': pixel_values
            }
        except Exception as e:
            return self._create_empty_sample()

    # ======================================== DPO数据处理 ========================================

    def _generate_negative_response(self, positive_response: str) -> str:
        # 生成负例响应: 从四个策略中随机选择一个
        # 由于选择的样本是正例，所以这里的策略是为了生成一个与正例不相关的负例
        strategies = [
            self._shuffle,
            self._prefix,
            self._deletion,
            self._fake
        ]
        return random.choice(strategies)(positive_response)

    def _shuffle(self, text: str) -> str:
        # 策略1：打乱词语顺序
        words = jieba.lcut(text)
        if len(words) <= 3:
            return "内容不相关"
        random.shuffle(words)
        return "".join(words)

    def _prefix(self, text: str) -> str:
        # 策略2：增加矛盾前缀
        prefixes = [
            "实际上情况相反：",
            "这个说法是错误的：",
            "以下并非事实："
        ]
        return random.choice(prefixes) + text

    def _deletion(self, text: str) -> str:
        # 策略3：随机删除部分文本
        words = text.split()
        if len(words) <= 3:
            return "信息不全"
        del_len = random.randint(1, len(words)//2)
        start = random.randint(0, len(words)-del_len)
        return "".join(words[:start] + words[start+del_len:])

    def _fake(self, text: str) -> str:
        # 策略4：自己制定错误事实
        false_text = [
            "无法识别图片。",
            "这个问题需要更多信息。"
        ]
        return random.choice(false_text)

    def generate_dpo_pairs(self, output_path: str) -> None:
        # 生成DPO训练对并保存
        samples = self._load_sft_samples()
        dpo_data = []
        
        for sample in samples:
            if len(sample['conversations']) < 2:
                continue
                
            query = sample['conversations'][0]['value']
            positive = sample['conversations'][1]['value']
            negative = self._generate_negative_response(positive)
            
            dpo_data.append({
                'image': sample['image'],
                'query': query,
                'positive': positive,
                'negative': negative,
                'id': sample.get('id', '')
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, ensure_ascii=False, indent=2)

    def _process_dpo_sample(self, sample: Dict) -> Dict:
        # 处理单个DPO样本
        try:
            image = Image.open(os.path.join(self.config['img_path'], sample['image'])).convert('RGB')
            pixel_values = self.visual_processor(images=image, return_tensors='pt')['pixel_values']
            
            return {
                'query': sample['query'],
                'positive': sample['positive'],
                'negative': sample['negative'],
                'pixel_values': pixel_values
            }
        except Exception as e:
            default=self._create_empty_sample()
            return {
                'query': default['input_ids'],
                'positive': default['labels'],
                'negative': 'It is a valid image, but the text data is missing.',
                'pixel_values': default['pixel_values']
            }

    # ================================================== 创建空图片样本数据 ========================================
    def _create_empty_sample(self) -> Dict:
        # 空样本用于错误处理
        empty_image = Image.new('RGB', (224, 224), color='white')
        # 构建对话模板
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the image about?\n<image>"},
           ]
        text = self.text_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            ).replace('<image>', self.config['img_pad_token'] * self.config['img_pad_num'])
        ans = 'It is an empty image'+self.text_tokenizer.eos_token
        # 标记化处理
        text_ids = self.text_tokenizer(text, return_tensors='pt')['input_ids']
        ans_ids = self.text_tokenizer(ans, return_tensors='pt')['input_ids']
        total_ids = text_ids+ans_ids
        labels = [self.text_tokenizer.pad_token_id]*len(text_ids)+ans_ids

        return {
            'input_ids': total_ids[:-1],
            'labels': labels[1:],
            'pixel_values': self.visual_processor(images=empty_image, return_tensors='pt')['pixel_values'][0]
        }

class UnifiedDataset(Dataset):
    # 数据集类
    def __init__(self, preprocessor: MultiModalDataPreprocessor, data_type: str):
        """
        Args:
            data_type: 'pretrain' | 'sft' | 'dpo'
        """
        self.preprocessor = preprocessor
        self.data_type = data_type
        
        if data_type == 'pretrain':
            self.raw_data = preprocessor._load_pretrain_samples()
        elif data_type == 'sft':
            self.raw_data = preprocessor._load_sft_samples()
        elif data_type == 'dpo':
            self.raw_data = preprocessor._load_sft_samples()

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        if self.data_type == 'pretrain':
            return self.preprocessor._process_pretrain_sample(self.raw_data[idx])
        elif self.data_type == 'sft':
            return self.preprocessor._process_sft_sample(self.raw_data[idx])
        else:
            return self.preprocessor._process_dpo_sample(self.raw_data[idx])


if __name__ == "__main__":

    """
    这里的数据集及模型下载地址：
    1. 预训练的图片数据集：https://hf-mirror.com/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip
    2. 预训练的文本数据集：https://hf-mirror.com/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions/blob/main/LLaVA-CC3M-Pretrain-595K/chat-translated.json
    3. sft的图片数据集：https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/blob/main/sft_images.zip
    4. sft的文本数据集：https://hf-mirror.com/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions/blob/main/LLaVA-Instruct-150K/translated/llava_instruct_150k.json
    5. dpo使用sft的数据集进行负例构建
    6. 大语言模型：https://hf-mirror.com/Qwen/Qwen2.5-1.5B-Instruct
    7. 视觉模型：https://hf-mirror.com/google/siglip-base-patch16-224
    """
    # 实例化
    processor = MultiModalDataPreprocessor(config)
    
    # 得到DPO数据
    processor.generate_dpo_pairs('/home/featurize/data/multimodal/dpo/dpo_gen.json')
    
    # 数据集
    # pretrain_dataset = UnifiedDataset(processor, 'pretrain')
    # sft_dataset = UnifiedDataset(processor, 'sft')
    # dpo_dataset = UnifiedDataset(processor, 'dpo')

