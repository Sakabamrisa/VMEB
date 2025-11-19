from transformers import AutoProcessor
from .MultimodalTrainer import *


class VideoMultimodalTrainer(MultimodalTrainer):
    def __init__(self, config, train_set, val_set):
        super().__init__(config, train_set, val_set)
        # 加载Qwen2.5-VL处理器
        self.processor = AutoProcessor.from_pretrained(config.name)
        if config.model_class == "Qwen2_5_VLForConditionalGeneration":
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info

    def edit_step(self, batch, training: bool):
        """
        重写edit_step方法来适应Qwen2.5-VL视频处理
        """
        self.model.train(training)
        self.original_model.train(training)

        # 处理基础模型输出（保持无梯度）
        with torch.no_grad():
            base_outputs = self.model(batch["loc"])
            torch.cuda.empty_cache()
            if not isinstance(base_outputs, torch.Tensor):
                base_logits = base_outputs.logits
            else:
                base_logits = base_outputs

            base_image_outputs = self.model(batch["loc_video"])
            torch.cuda.empty_cache()
            if not isinstance(base_image_outputs, torch.Tensor):
                base_image_logits = base_image_outputs.logits
            else:
                base_image_logits = base_image_outputs

        # 执行模型编辑
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], self.processor, batch["cond"])
        edit_time = time.time() - start

        l_total, l_edit, l_loc, l_base = 0, 0, 0, 0
        info_dict = {}

        # 计算各种损失
        with torch.set_grad_enabled(training):
            # 编辑损失 - 原始视频，新问题
            post_edit_outputs = edited_model(batch["edit_outer"])
            torch.cuda.empty_cache()
            post_batch_labels = batch["edit_outer"]["labels"]
            if not isinstance(post_edit_outputs, torch.Tensor):
                post_edit_logits = post_edit_outputs.logits
            else:
                post_edit_logits = post_edit_outputs

            # 编辑损失 - 重新描述的视频
            post_image_edit_outputs = edited_model(batch["edit_outer_video"])
            torch.cuda.empty_cache()
            post_image_batch_labels = batch["edit_outer_video"]["labels"]
            if not isinstance(post_image_edit_outputs, torch.Tensor):
                post_image_edit_logits = post_image_edit_outputs.logits
            else:
                post_image_edit_logits = post_image_edit_outputs

            # 内部编辑损失
            inner_edit_outputs = edited_model(batch["edit_inner"])
            torch.cuda.empty_cache()
            inner_batch_labels = batch["edit_inner"]["labels"]
            if not isinstance(inner_edit_outputs, torch.Tensor):
                inner_edit_logits = inner_edit_outputs.logits
            else:
                inner_edit_logits = inner_edit_outputs

            # 计算各类损失，处理形状不匹配问题
            if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)["nll"]
            else:
                l_edit = self.model.edit_loss_fn(self.config, post_edit_logits,post_batch_labels[:, -post_edit_logits.shape[1] - 1:])["nll"]
            if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:
                l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)["nll"]
            else:
                l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1] - 1:])["nll"]
                # 收集评估指标
            with torch.no_grad():
                if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)
                else:
                    post_edit_dict = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1] - 1:])

                if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits, inner_batch_labels)
                else:
                    inner_edit_dict = self.model.edit_loss_fn(self.config, inner_edit_logits,
                                                              inner_batch_labels[:, -inner_edit_logits.shape[1] - 1:])

                if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:
                    image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits,post_image_batch_labels)
                else:
                    image_rephrase_edit_dict = self.model.edit_loss_fn(self.config, post_image_edit_logits,post_image_batch_labels[:, -post_image_edit_logits.shape[1] - 1:])

            # 计算局部性损失
            post_base_outputs = edited_model(batch["loc"])
            torch.cuda.empty_cache()
            if not isinstance(post_base_outputs, torch.Tensor):
                post_base_logits = post_base_outputs.logits
                kl_mask = post_base_outputs.attention_mask
            else:
                post_base_logits = post_base_outputs
                kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

            post_image_base_outputs = edited_model(batch["loc_video"])
            torch.cuda.empty_cache()
            if not isinstance(post_image_base_outputs, torch.Tensor):
                post_image_base_logits = post_image_base_outputs.logits
                kl_image_mask = post_image_base_outputs.attention_mask
            else:
                post_image_base_logits = post_image_base_outputs
                kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(
                    post_image_base_logits.device)

            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
            l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

        # 检查NaN损失
        if l_edit.isnan():
            print("l_edit is nan")
            print("input: ", batch["edit_outer"]['text_input'])
        elif l_image_edit.isnan():
            print("l_video_edit is nan")
            print("input: ", batch["edit_outer_video"]['text_input'])
        elif l_loc.isnan():
            print("l_loc is nan")
            print("input: ", batch["loc"]['text_input'])
        elif l_image_loc.isnan():
            print("l_video_loc is nan")
            print("input: ", batch["loc_video"]['text_input'])

        # 计算总损失
        l_total_edit = self.config.cedit * l_edit + self.config.cloc * (
                    l_loc + l_image_loc) + self.config.iedit * l_image_edit

        # 反向传播
        if training and self.config.alg != 'ft':
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

        # 计算各种精度指标
        post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1,
                                                    dim=-1).indices
        base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

        post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1),
                                                          k=10, dim=-1).indices
        base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10,
                                                     dim=-1).indices

        # 收集信息字典
        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/video_edit'] = l_image_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict['edit/acc'] = post_edit_dict["acc"].item()
        info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
        info_dict['edit/prob'] = post_edit_dict["prob"].item()
        info_dict['inner/acc'] = inner_edit_dict["acc"].item()
        info_dict['video_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
        info_dict["time/edit"] = edit_time
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1)) / \
                               post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["video_loc/acc"] = sum(
            post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1)) / \
                                     post_image_base_logits_softmax_top_k.view(-1).shape[0]

        l_base = torch.tensor(0.0)
        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()

        # 合并模型信息
        info_dict = {**info_dict, **model_info}

        # print("总显存占用：")
        # print(torch.cuda.memory_summary())  # 观察 Reserved/Allocated 比例
        torch.cuda.empty_cache()

        return l_total, l_edit, l_loc, l_base, info_dict