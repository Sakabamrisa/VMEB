import json
import typing
import os
import torch
import numpy as np
import av
import logging
from PIL import Image
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
import transformers
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from ..trainer.utils import dict_to
from .processor.base_dataset import BaseDataset

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoCaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, size: typing.Optional[int] = None, config=None, hop=None, *args, **kwargs):
        # 使用LLaVA-NeXT-Video处理器（包含视觉和文本处理）
        if config.model_class == "LlavaNextVideoForConditionalGeneration":
            processor = transformers.LlavaNextVideoProcessor.from_pretrained(config.name)
            vis_processor = processor  # 包含视频和文本处理
            tokenizer = processor.tokenizer  # 从处理器获取分词器
            self.num_frames = config.num_frames if hasattr(config, 'num_frames') else 8
            self.target_size = (28, 28)  # 默认目标帧大小
            self.read_video_pyav = self._create_video_reader()  # 创建视频读取器
            SYSTEM_PROMPT = """You are a professional assistant, and now you need to answer a question. The question contains options which you should choose from, for example:
            question: "What is the action performed by the person in the video?" bathing, watering, washing, bubbling
            ASSISTANT: bathing
            Please strictly answer according to the example format, only output the answer, do not add explanations.Answer without quotes.
            question:
            """

            logger.info(f"已加载 LLaVA-NeXT-Video 处理器和分词器: {config.name}")
        elif config.model_class == "Qwen2_5_VLForConditionalGeneration":
            processor = AutoProcessor.from_pretrained(config.name)
            vis_processor = processor
            tokenizer = processor.tokenizer

            logger.info(f"已加载 Qwen2.5-VL 处理器和分词器: {config.name}")
        elif config.model_class == "LlavaOnevisionForConditionalGeneration":
            processor = AutoProcessor.from_pretrained(config.name)
            vis_processor = processor  # 包含视频和文本处理
            tokenizer = processor.tokenizer  # 从处理器获取分词器
            self.num_frames = config.num_frames if hasattr(config, 'num_frames') else 8
            self.target_size = (28, 28)  # 默认目标帧大小
            self.read_video_pyav = self._create_video_reader()  # 创建视频读取器
            SYSTEM_PROMPT = """You are a professional assistant, and now you need to answer a question. The question contains options which you should choose from, for example:
            question: "What is the action performed by the person in the video?" bathing, watering, washing, bubbling
            ASSISTANT: bathing
            Please strictly answer according to the example format, only output the answer, do not add explanations.Answer without quotes.
            question:
            """

            logger.info(f"已加载 Video-UTR 处理器和分词器: {config.name}")
        else:
            raise NotImplementedError(f"不支持的模型类别: {config.model_class}")

        vis_root = config.coco_video  # 视频路径
        rephrase_root = config.rephrase_video  # 重述视频路径

        # 验证目录存在
        if not os.path.exists(vis_root):
            logger.warning(f"视频根目录不存在: {vis_root}")
        if not os.path.exists(rephrase_root):
            logger.warning(f"重述视频根目录不存在: {rephrase_root}")

        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32

        data = []
        if size is not None:
            self.annotation = self.annotation[:size]

        if hop:
            self.hop = hop
            assert int(hop) in [1, 2, 3, 4], "hop应为1、2、3或4"
            port_types = ['', '1-hop', '2-hop', '3-hop', '4-hop']
            port_type = port_types[int(hop)]

        skipped_count = 0
        for record in tqdm(self.annotation, ncols=120, desc='加载数据'):

            if record['alt'] == "":
                skipped_count += 1
                continue

            if hop and 'port_new' not in record.keys():
                skipped_count += 1
                continue

            # 构建完整的视频路径
            video_path = os.path.join(self.vis_root, record["video"])
            rephrase_video_path = os.path.join(self.rephrase_root, record["video_rephrase"])
            locality_video_path = os.path.join(self.vis_root, record['m_loc'])

            # 验证文件存在（但不阻止加载，只记录警告）
            for path, desc in [(video_path, "主视频"),
                               (rephrase_video_path, "重述视频"),
                               (locality_video_path, "局部性视频")]:
                if not os.path.exists(path):
                    logger.warning(f"{desc}不存在: {path}")

            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'video': video_path,
                'video_rephrase': rephrase_video_path,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }

            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            item['multimodal_locality_video'] = locality_video_path

            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']

            if hop and 'port_new' in record.keys():
                item['portability_prompt'] = []
                item['portability_ground_truth'] = []
                find_hop = False
                for ports in record['port_new']:
                    if ports['port_type'] == port_type:
                        find_hop = True
                        port_q = ports['Q&A']['Question']
                        port_a = ports['Q&A']['Answer']
                        item['portability_prompt'].append(port_q)
                        item['portability_ground_truth'].append(port_a)
                        break

                if not find_hop:
                    skipped_count += 1
                    continue

            data.append(item)

        logger.info(f"加载了 {len(data)} 个样本，跳过了 {skipped_count} 个样本")
        self._data = data

    def build_prompt(self, question, answer, no_video = False):
        if self.config.model_class == "Qwen2_5_VLForConditionalGeneration":
        # Qwen2.5-VL的prompt构建函数
            START_TOKEN = "<|im_start|>"
            END_TOKEN = "<|im_end|>"
            VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"
            prompt = f'{START_TOKEN}system\nYou are a helpful assistant.{END_TOKEN}\n{START_TOKEN}user\n{VIDEO_TOKEN}{question}{END_TOKEN}\n{START_TOKEN}assistant\n{answer}'
            return prompt
        elif self.config.model_class == "LlavaNextVideoForConditionalGeneration" or "LlavaOnevisionForConditionalGeneration":
            VIDEO_TOKEN = "<video>"
            if no_video == True :
                return f"USER: {question} ASSISTANT: {answer}"
            else:
                return f"USER: {VIDEO_TOKEN}\n{question} ASSISTANT: {answer}"

    def _create_video_reader(self):
        """LLaVA-NEXT-Video的视频读取器"""

        def read_video_pyav(container, indices):
            """
            使用PyAV解码视频帧。

            Args:
                container (`av.container.input.InputContainer`): PyAV容器。
                indices (`List[int]`): 要解码的帧索引列表。

            Returns:
                result (np.ndarray): 形状为(num_frames, height, width, 3)的解码帧数组。
            """
            frames = []
            container.seek(0)
            start_index = indices[0]
            end_index = indices[-1]
            for i, frame in enumerate(container.decode(video=0)):
                if i > end_index:
                    break
                if i >= start_index and i in indices:
                    frames.append(frame)
            return np.stack([x.to_ndarray(format="rgb24") for x in frames]) if frames else None

        # 创建一个包装函数，负责处理视频路径、计算索引和错误处理
        def video_loader(video_path, num_frames=self.num_frames):
            try:
                # 检查文件是否存在
                if not os.path.exists(video_path):
                    logger.warning(f"视频文件不存在: {video_path}")
                    return np.zeros((num_frames, *self.target_size, 3), dtype=np.uint8)

                # 打开视频容器
                container = av.open(video_path)

                # 获取视频流信息
                video_stream = container.streams.video[0]
                total_frames = video_stream.frames

                # 处理可能的无效帧数情况
                if total_frames <= 0 or total_frames is None:
                    # 估计总帧数：持续时间（秒）* 帧率
                    fps = video_stream.average_rate
                    duration = float(container.duration) / av.time_base
                    total_frames = max(int(duration * fps), 1)  # 确保至少有1帧

                # 计算均匀采样的帧索引
                indices = np.arange(0, total_frames, total_frames / 4).astype(int)

                # 使用原始函数读取帧
                frames = read_video_pyav(container, indices)

                # 检查是否成功读取了帧
                if frames is None or len(frames) == 0:
                    logger.warning(f"无法从视频中提取帧: {video_path}")
                    return np.zeros((num_frames, *self.target_size, 3), dtype=np.uint8)

                # 确保所有帧具有相同的尺寸
                # 如果帧大小不一致，调整为目标尺寸
                if any(frame.shape[:2] != self.target_size for frame in frames):
                    from PIL import Image
                    resized_frames = []
                    for frame in frames:
                        img = Image.fromarray(frame)
                        img = img.resize(self.target_size, Image.LANCZOS)
                        resized_frames.append(np.array(img))
                    frames = np.stack(resized_frames)

                return frames

            except Exception as e:
                logger.error(f"读取视频错误 {video_path}: {e}")
                # 创建空白帧序列作为后备方案
                return np.zeros((num_frames, *self.target_size, 3), dtype=np.uint8)

        return video_loader

    def __getitem__(self, index):
        """获取指定索引的数据样本"""
        data = deepcopy(self._data[index])

        # 加载视频路径
        video_path = data['video']
        rephrase_video_path = data['video_rephrase']
        locality_video_path = data['multimodal_locality_video']

        if self.config.model_class == "Qwen2_5_VLForConditionalGeneration":
            image_inputs = None
            video_inputs = None
            video_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video",
                         "video": video_path,
                         "min_pixels": 48 * 28 * 28,
                         "max_pixels": 128 * 28 * 28,
                         "min_frames": 4,
                         "max_frames": 16,
                         "total_pixels": 128 * 28 * 28,
                         "fps": 2},

                    ],
                }
            ]
            image_inputs, video_inputs = process_vision_info(video_messages)
            data['image'] = image_inputs
            data['video'] = video_inputs
            torch.cuda.empty_cache()  # 在关键步骤后手动调用

            rephrase_image_inputs = None
            rephrase_video_inputs = None
            rephrase_video_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video",
                         "video": rephrase_video_path,
                         "min_pixels": 48 * 28 * 28,
                         "max_pixels": 128 * 28 * 28,
                         "min_frames": 4,
                         "max_frames": 16,
                         "total_pixels": 128 * 28 * 28,
                         "fps": 2},

                    ],
                }
            ]
            rephrase_image_inputs, rephrase_video_inputs = process_vision_info(rephrase_video_messages)
            data['rephrase_image'] = rephrase_image_inputs
            data['rephrase_video'] = rephrase_video_inputs
            torch.cuda.empty_cache()  # 在关键步骤后手动调用

            multimodal_locality_image_inputs = None
            multimodal_locality_video_inputs = None
            multimodal_locality_video_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video",
                         "video": locality_video_path,
                         "min_pixels": 48 * 28 * 28,
                         "max_pixels": 128 * 28 * 28,
                         "min_frames": 4,
                         "max_frames": 16,
                         "total_pixels": 128 * 28 * 28,
                         "fps": 2},

                    ],
                }
            ]
            multimodal_locality_image_inputs, multimodal_locality_video_inputs = process_vision_info(multimodal_locality_video_messages)
            data['multimodal_locality_image'] = multimodal_locality_image_inputs
            data['multimodal_locality_video'] = multimodal_locality_video_inputs
            torch.cuda.empty_cache()  # 在关键步骤后手动调用

            return data


        elif self.config.model_class == "LlavaNextVideoForConditionalGeneration" or "LlavaOnevisionForConditionalGeneration":
            # 读取视频帧
            video_frames = self.read_video_pyav(video_path, self.num_frames)
            rephrase_video_frames = self.read_video_pyav(rephrase_video_path, self.num_frames)
            locality_video_frames = self.read_video_pyav(locality_video_path, self.num_frames)

            # 存储视频帧
            data['video'] = video_frames
            data['video_rephrase'] = rephrase_video_frames
            data['multimodal_locality_video'] = locality_video_frames
            data['image'] = None
            data['image_rephrase'] = None
            data['multimodal_locality_image'] = None

            # 保存视频路径供后续使用
            data['video_path'] = video_path
            data['video_rephrase_path'] = rephrase_video_path
            data['multimodal_locality_video_path'] = locality_video_path

            return data

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self._data)

    def collate_fn(self, batch):
        # --- 提取原始数据 ---
        src = [b['prompt'] for b in batch] # 原始问题列表
        trg = [b['target'] for b in batch] # 原始答案列表
        cond = [b['cond'] for b in batch] if 'cond' in batch[0] else None # 条件信息
        rephrase = [b['rephrase_prompt'] for b in batch] # 重述问题列表
        loc_q = [b['locality_prompt'] for b in batch] # 文本局部性问题
        loc_a = [b['locality_ground_truth'] for b in batch] # 文本局部性答案
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch] # 多模态局部性问题
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch] # 多模态局部性答案
        if self.config.model_class == "Qwen2_5_VLForConditionalGeneration":
            image = [b['image'] for b in batch]  # 原始图像 (可能为 None 或列表)
            image_rephrase = [b['rephrase_image'] for b in batch]  # 重述图像
            m_loc_image = [b['multimodal_locality_image'] for b in batch]  # 多模态局部性图像
            video = [b['video'] for b in batch]  # 原始视频 (可能为 None 或列表)
            video_rephrase = [b['rephrase_video'] for b in batch]  # 重述视频
            m_loc_video = [b['multimodal_locality_video'] for b in batch]  # 多模态局部性视频
        elif self.config.model_class == "LlavaNextVideoForConditionalGeneration" or "LlavaOnevisionForConditionalGeneration":
            video = [b['video'] for b in batch]
            video_rephrase = [b['video_rephrase'] for b in batch]
            m_loc_video = [b['multimodal_locality_video'] for b in batch]

        if self.config.model_class == "Qwen2_5_VLForConditionalGeneration":
            fps = 1.0
            max_frames = 16  # 限制最大帧数
            inner_prompt_example = self.build_prompt(src[0], trg[0]) # 仅用第一个样本构建示例 prompt
            edit_inner = {}
            edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)] # 问题 + 答案 字符串 (可能用于评估?)
            edit_inner['question_text'] = src # <--- 新增：存储原始问题列表
            # 使用 processor 处理 (假设 processor 能处理批量的 text, image, video)
            # 注意：processor 的 text 参数应该是一个列表，对应批处理
            # 如果 processor 不能直接处理批量 text/image/video，需要循环处理或调整
            edit_inner['inputs'] = self.vis_processor(
                text=[self.build_prompt(s, t) for s, t in zip(src, trg)], # 为批次中每个样本构建prompt
                images=image[0], # 传递图像列表 (如果 processor 支持)
                videos=video[0], # 传递视频列表 (如果 processor 支持)
                fps=fps,
                padding=True, return_tensors='pt', max_frames=max_frames
            ).to(self.config.device)
            edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src] # 原始问题长度
            edit_inner['labels'] = self.tok(trg, padding=True, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors="pt")["input_ids"] # 答案 token

            torch.cuda.empty_cache()

            # --- 处理 edit_outer ---
            outer_prompt_example = self.build_prompt(rephrase[0], trg[0])
            edit_outer = {}
            edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
            edit_outer['question_text'] = rephrase # <--- 新增：存储重述的问题列表
            edit_outer['inputs'] = self.vis_processor(
                text=[self.build_prompt(r, t) for r, t in zip(rephrase, trg)],
                images=image[0], # 使用原始图像
                videos=video[0], # 使用原始视频
                fps=fps,
                padding=True, return_tensors='pt', max_frames=max_frames
            ).to(self.config.device)
            edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, padding=True, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors="pt")["input_ids"]

            torch.cuda.empty_cache()

            # --- 处理 edit_outer_video ---
            outer_video_prompt_example = self.build_prompt(src[0], trg[0])
            edit_outer_video = {}
            edit_outer_video['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
            edit_outer_video['question_text'] = src # <--- 新增：存储原始问题列表
            edit_outer_video['inputs'] = self.vis_processor(
                text=[self.build_prompt(s, t) for s, t in zip(src, trg)],
                images=image_rephrase[0], # 使用重述图像
                videos=video_rephrase[0], # 使用重述视频
                fps=fps, padding=True,
                return_tensors='pt', max_frames=max_frames
            ).to(self.config.device)
            edit_outer_video['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_outer_video['labels'] = self.tok(trg, padding=True, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors="pt")["input_ids"]

            torch.cuda.empty_cache()

            # --- 处理 loc (文本局部性) ---
            loc_prompt_example = self.build_prompt(loc_q[0], loc_a[0])
            loc = {}
            loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
            loc['question_text'] = loc_q # <--- 新增：存储局部性问题列表
            loc['inputs'] = self.vis_processor(
                text=[self.build_prompt(q, a) for q, a in zip(loc_q, loc_a)],
                images=None, # 无图像
                videos=None, # 无视频
                padding=True,
                return_tensors='pt', max_frames=max_frames # max_frames 在无视频时可能无效
            ).to(self.config.device)
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, padding=True, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors="pt")["input_ids"]

            torch.cuda.empty_cache()

            # --- 处理 loc_video (多模态局部性) ---
            m_loc_prompt_example = self.build_prompt(m_loc_q[0], m_loc_a[0])
            loc_video = {}
            loc_video['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
            loc_video['question_text'] = m_loc_q # <--- 新增：存储多模态局部性问题列表
            loc_video['inputs'] = self.vis_processor(
                text=[self.build_prompt(q, a) for q, a in zip(m_loc_q, m_loc_a)],
                images=m_loc_image[0], # 使用多模态局部性图像
                videos=m_loc_video[0], # 使用多模态局部性视频
                fps=fps, padding=True, return_tensors='pt', max_frames=max_frames
            ).to(self.config.device)
            loc_video['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
            loc_video['labels'] = self.tok(m_loc_a, padding=True, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors="pt")["input_ids"]

            torch.cuda.empty_cache()

            # --- 处理条件信息 ---
            if cond is not None:
                cond = self.tok(
                    cond,
                    return_tensors="pt",
                    padding=True,
                    max_length=self.max_length,
                    truncation=True,
                )
            else:
                cond = None # 保持 None 如果原始 cond 是 None

            # --- 处理可移植性测试 ---
            edit_ports = None
            # 检查批次中第一个样本是否有 portability_prompt 键
            if 'portability_prompt' in batch[0] and batch[0]['portability_prompt'] is not None:
                edit_ports = []
                # 假设 portability 数据结构在批次内是一致的
                # 注意：原始代码似乎只处理了 batch[0] 的 portability，这里我们假设需要处理整个批次
                # 如果 portability 数据结构复杂，这里的处理需要调整
                port_qs = [item for b in batch for item in b.get('portability_prompt', [])] # 收集所有 portability 问题
                port_as = [item for b in batch for item in b.get('portability_ground_truth', [])] # 收集所有 portability 答案

                if port_qs and port_as: # 确保有数据
                    port_prompt_examples = [self.build_prompt(q, a) for q, a in zip(port_qs, port_as)]
                    port = {}
                    port['text_input'] = [" ".join([q, a]) for q, a in zip(port_qs, port_as)]
                    port['question_text'] = port_qs # <--- 新增：存储可移植性问题列表
                    # 假设可移植性测试使用 edit_inner 的图像/视频
                    port['inputs'] = self.vis_processor(
                        text=port_prompt_examples,
                        images=image[0], # 重复使用原始图像/视频 (需要确认逻辑)
                        videos=video[0],
                        fps=fps,
                        padding=True, return_tensors='pt', max_frames=max_frames
                    ).to(self.config.device)
                    port['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in port_qs]
                    port['labels'] = self.tok(port_as, padding=True, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors="pt")["input_ids"]
                    # 注意：这里将所有 port 合并到一个字典处理了，如果需要分开处理每个 port，需要修改
                    edit_ports = port # 或者 append(port) 如果需要列表
                    torch.cuda.empty_cache()


            # --- 最终批次组装 ---
            batch_output = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "edit_outer_video": edit_outer_video,
                "loc": loc,
                "loc_video": loc_video,
                'port': edit_ports, # 可能是 None 或包含 port 数据的字典
                "cond": cond # 可能是 None 或包含 cond 数据的字典
            }

            # --- 移动到设备 ---
            # 确保 dict_to 函数存在且能正确处理嵌套字典和 None 值
            return dict_to(batch_output, self.config.device)

        elif self.config.model_class == "LlavaNextVideoForConditionalGeneration" or "LlavaOnevisionForConditionalGeneration":
            SYSTEM_PROMPT = """You are a professional assistant, and now you need to answer a question. The question contains options which you should choose from, for example:
            question: "What is the action performed by the person in the video?" bathing, watering, washing, bubbling
            ASSISTANT: bathing
            Please strictly answer according to the example format, only output the answer, do not add explanations.Answer without quotes.
            question:
            """
            # edit_inner - 主视频，原始问题
            edit_inner = {}
            edit_inner['video'] = self.vis_processor(text='', videos=video[0], return_tensors='pt')[
                'pixel_values_videos'].to(self.config.device)  # torch.stack(video, dim=0)
            edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
            edit_inner['inputs'] = self.vis_processor(text=self.build_prompt(SYSTEM_PROMPT + src[0], trg[0]), padding = True,
                                                      videos=video[0], return_tensors='pt').to(self.config.device)
            edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt", )["input_ids"]

            torch.cuda.empty_cache()

            # edit_outer - 主视频，重述问题
            edit_outer = {}
            edit_outer['video'] = self.vis_processor(text='', videos=video[0], return_tensors='pt')[
                'pixel_values_videos'].to(self.config.device)  # torch.stack(video, dim=0)
            edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
            edit_outer['inputs'] = self.vis_processor(
                text=self.build_prompt(SYSTEM_PROMPT + rephrase[0], trg[0]), videos=video[0], padding = True,
                return_tensors='pt').to(self.config.device)
            edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt", )["input_ids"]

            torch.cuda.empty_cache()

            # edit_outer_video - 重述视频，原始问题
            edit_outer_video = {}
            edit_outer_video['video'] = self.vis_processor(text='', videos=video_rephrase[0], return_tensors='pt')[
                'pixel_values_videos'].to(self.config.device)  # torch.stack(video_rephrase, dim=0)
            edit_outer_video['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
            edit_outer_video['inputs'] = self.vis_processor(
                text=self.build_prompt(SYSTEM_PROMPT + src[0], trg[0]), videos=video_rephrase[0], padding = True,
                return_tensors='pt').to(self.config.device)
            edit_outer_video['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            edit_outer_video['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt", )["input_ids"]

            torch.cuda.empty_cache()

            # loc - 文本局部性测试（无视频）
            loc = {}
            loc['video'] = None  # torch.zeros(1, 3, *self.target_size)  # 占位符
            loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
            loc['inputs'] = self.vis_processor(text=self.build_prompt(loc_q[0], loc_a[0], True), videos=None, padding = True,
                                               return_tensors='pt').to(self.config.device)
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt", )["input_ids"]

            torch.cuda.empty_cache()

            # m_loc - 多模态局部性测试（视频）
            loc_video = {}
            loc_video['video'] = self.vis_processor(text='', videos=m_loc_video[0], return_tensors='pt')[
                'pixel_values_videos'].to(self.config.device)  # torch.stack(m_loc_video, dim=0)
            loc_video['text_input'] = [" ".join([q, a]) for q, a in zip(m_loc_q, m_loc_a)]
            loc_video['inputs'] = self.vis_processor(
                text=self.build_prompt(SYSTEM_PROMPT + m_loc_q[0], m_loc_a[0]), videos=m_loc_video[0], padding = True,
                return_tensors='pt').to(self.config.device)
            loc_video['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in m_loc_q]
            loc_video['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt", )["input_ids"]

            torch.cuda.empty_cache()

            # 条件信息处理
            cond = self.tok(
                cond,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )

            # 可移植性测试部分
            edit_ports = None
            if 'portability_prompt' in batch[0].keys():
                edit_ports = []
                for port_q, port_a in zip(batch[0]['portability_prompt'], batch[0]['portability_ground_truth']):
                    port = {}
                    port['video'] = torch.stack(video, dim=0)
                    port['text_input'] = [' '.join([port_q, port_a])]
                    port['inputs'] = self.vis_processor(
                        text=self.build_prompt(SYSTEM_PROMPT + port_q[0], port_a[0]), videos=video[0], padding = True,
                        return_tensors='pt').to(self.config.device)
                    port['prompts_len'] = [len(self.tok.encode(port_q, add_special_tokens=False))]
                    port['labels'] = self.tok([port_a], add_special_tokens=False, return_tensors="pt", )["input_ids"]
                    edit_ports.append(port)

            torch.cuda.empty_cache()

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "edit_outer_video": edit_outer_video,
                "loc": loc,
                "loc_video": loc_video,
                'port': edit_ports,
                "cond": cond
            }

            torch.cuda.empty_cache()

            # 移动到指定设备
            return dict_to(batch, self.config.device)