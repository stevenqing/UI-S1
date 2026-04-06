

import copy
import logging
import os
import random
import re
import time
import traceback
import uuid
from collections import defaultdict
from typing import Any, List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from x.data.agent.json import JsonFormat
from x.data.agent.sftv2 import SFTv2Format
from x.io import JsonWrap
from x.parallel.parallel_task import ParallelTask
from x.qwen.data_format import slim_messages

import verl.utils.torch_functional as verl_F
from verl.protocol import DataProto
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask


class QwenMessages2Inputs():
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DictConfig, processor: Any | None = None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
      
        self.max_pixels = 12800*28*28
        self.min_pixels = 4*28*28
        self.num_image_limit = config.get("num_image_limit", 2)

        self.max_prompt_length = config.get("max_prompt_length", 32768)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)

    def __call__(self, state):
        messages = state['messages']
        check_options = state['check_options']
        row_dict = {}
        messages = slim_messages(messages, num_image_limit=self.num_image_limit)
        last_image_ele = None
        for msg in messages:
            for content in msg['content']:
                # Very Important
                if 'image' in content:
                    if 'min_pixels' not in content: # TODO fix bug, respect to the resized height
                        content['min_pixels'] = self.min_pixels
                    if 'max_pixels' not in content:
                        content['max_pixels'] = self.max_pixels
                    last_image_ele = content
        assert messages[-1]['role'] == 'user'

        assert self.processor is not None
        from verl.utils.dataset.vision_utils import (process_image,
                                                     process_video)

        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        multi_modal_data = {}
        image_inputs, video_inputs = process_vision_info(messages)
        assert 0 < len(image_inputs)<=self.num_image_limit
        
        width, height = last_image_ele['width'], last_image_ele['height']
        resized_width, resized_height = image_inputs[-1].size
        
        model_inputs = self.processor(text=[raw_prompt], images=image_inputs, videos=video_inputs, return_tensors="pt")
        if image_inputs is not None:
            # Use integer division to match vision encoder's calculation
            # Allow small tolerance for edge cases in batch processing
            expected_tokens = sum((_.size[0]//28) * (_.size[1]//28) for _ in image_inputs)
            actual_tokens = (model_inputs['input_ids'] == 151655).sum().item()
            if abs(expected_tokens - actual_tokens) > 5:
                raise AssertionError(
                    f"Image token count mismatch: expected ~{expected_tokens}, got {actual_tokens}. "
                    f"Images: {[_.size for _ in image_inputs]}"
                )

        multi_modal_data = {
            'image': image_inputs
        }
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        if "second_per_grid_ts" in model_inputs:
            model_inputs.pop("second_per_grid_ts")

        # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
        
        row_dict["multi_modal_data"] = multi_modal_data
        row_dict["multi_modal_inputs"] = dict(model_inputs)

        # second_per_grid_ts isn't used for training, just for mrope
        row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)


        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        assert self.processor.image_processor.__class__.__name__ != "Qwen2_5VLImageProcessor"
        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        row_dict['reward_model'] = {
            "style": "rule",
            "ground_truth": check_options
        }

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = message_translate(messages, to_format="openai")

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
      
        row_dict["index"] = index
        if 'extra_info' not in row_dict:
            row_dict['extra_info'] = {}
        row_dict['extra_info']['resized_width'] = resized_width
        row_dict['extra_info']['resized_height'] = resized_height
        row_dict['extra_info']['width'] = width
        row_dict['extra_info']['height'] = height

        return row_dict



class StdTrajectory():
    def __init__(self, line,actions_only,hint,hindsight=False,explorer=False) -> None:
        self.line = line[()]
        self.num_steps = len(self.line['steps'])
        from x.data.agent.space.std_space import RAW_SPACE
        self.fm = SFTv2Format(RAW_SPACE)
        self.state = None
        self.hindsight = hindsight
        self.explorer = explorer

    def get_next(self, model_response):
        state = self.fm.gen_next_round(self.line, self.state, previous_model_response=model_response, hindsight=self.hindsight)
        if state is None:
            return "Finished"
        return state
 
class MultiRoundGenerator():
    def __init__(self, batch: DataProto, rollout_n, msg_man, patch_threshold=0,actions_only=None,hint=False,hindsight_fraction=0.0,explorer_fraction=0.0) -> None:
        self.rollout_n = rollout_n
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.non_tensor_batch["line"]))], dtype=object)

        repeat_batch = batch.repeat(repeat_times=self.rollout_n, interleave=True) # need set rollout kwargs to 1
        self.batch = repeat_batch
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(len(self.batch))], dtype=object)
        self.batch.non_tensor_batch["traj_uid"] = traj_uid
        self.task_queue = [StdTrajectory(line,actions_only,hint) for line in self.batch.non_tensor_batch["line"]]

        # Mark first N rollouts per K-group as hindsight
        n_hindsight = int(hindsight_fraction * rollout_n)
        for i in range(len(self.task_queue)):
            group_idx = i % rollout_n  # position within K-group
            if group_idx < n_hindsight:
                self.task_queue[i].hindsight = True

        # D4: Mark last N rollouts per K-group as explorers (suppress terminate)
        # Explorers take the LAST slots to avoid overlap with hindsight (first slots)
        n_explorer = int(explorer_fraction * rollout_n)
        for i in range(len(self.task_queue)):
            group_idx = i % rollout_n
            if group_idx >= rollout_n - n_explorer:
                self.task_queue[i].explorer = True

        self.finished = [False for i in range(len(self.task_queue))]
        self.current_response = [None for i in range(len(self.task_queue))]
        self.error_num = [0 for i in range(len(self.task_queue))]
        self.msg_man = msg_man
        from x.data.agent.space.std_space import RAW_SPACE
        self.fm = SFTv2Format(RAW_SPACE)
        self.patch_threshold = patch_threshold
        self.hint = hint
        self.hindsight_fraction = hindsight_fraction
        self.explorer_fraction = explorer_fraction
        print(f'Finish generator init (hindsight_fraction={hindsight_fraction}, n_hindsight_per_group={n_hindsight}, explorer_fraction={explorer_fraction}, n_explorer_per_group={n_explorer})')


    def _fetch_next(self, ptr):
        if self.finished[ptr]:
            return True, (None, None)
        current_gen = self.task_queue[ptr]
        current_response = self.current_response[ptr]
        state = current_gen.get_next(current_response)
        if state == "Finished":
            return True, ("Finished", state)
        row_dict = self.msg_man(state)
        row_dict['ptr'] = ptr
        return True, (row_dict, state)
        

    def fetch_batch(self):
        while True:
            batch = []

            tasks = list(range(len(self.task_queue)))
            # input()
            mid_result = ParallelTask((tasks,), self._fetch_next, total=len(tasks), num_process=len(tasks), passing_indices=False, return_list=True).run_and_collect(tqdm_args={"disable": False})
            assert len(mid_result) == len(self.task_queue)
            for ptr, res in enumerate(mid_result):
                row_dict, state = res
                if row_dict == None:
                    continue
                self.current_response[ptr]= None
                if row_dict == "Finished":
                    self.finished[ptr] = True
                else:
                    self.task_queue[ptr].state = state
                    row_dict['uid'] = self.batch.non_tensor_batch['uid'][ptr]
                    row_dict['traj_uid'] = self.batch.non_tensor_batch['traj_uid'][ptr]
                    row_dict['step_id'] = state['step_id']
                    if 'extra_info' in row_dict:
                        row_dict['extra_info']['step_num'] = state['step_id'] + 1  # 1-based
                    row_dict['execution_id'] = self.task_queue[ptr].line.get('execution_id', '')
                    if 'data_source' in self.batch.non_tensor_batch:
                        row_dict['data_source'] = self.batch.non_tensor_batch['data_source'][ptr]
                    else:
                        row_dict['data_source'] = self.task_queue[ptr].line.get('data_source', 'gui360')
                    row_dict['reward_model'] = {
                        "style": "rule",
                        "ground_truth": {
                            "check_options": state['check_options'],
                            'num_steps': self.task_queue[ptr].num_steps,
                            'thought': state['thought'],
                            }
                    }
                    row_dict['is_hindsight'] = self.task_queue[ptr].hindsight
                    row_dict['is_explorer'] = self.task_queue[ptr].explorer

                    # Store raw messages and next screenshot path for hindsight aux loss
                    # Wrap in dict to prevent numpy from creating 2D arrays
                    # (message lists have different lengths at different steps)
                    row_dict['raw_messages'] = {'msgs': copy.deepcopy(state['messages'])}
                    line = self.task_queue[ptr].line
                    step_id = state['step_id']
                    if step_id + 1 < len(line['steps']):
                        row_dict['next_screenshot_path'] = line['steps'][step_id + 1]['screenshot']
                    else:
                        row_dict['next_screenshot_path'] = None
                    # Store pre-computed pi_V description of s_{t+1} for dual-channel hindsight
                    if step_id < len(line['steps']):
                        row_dict['next_desc_t1'] = line['steps'][step_id].get('desc_t1', None)
                    else:
                        row_dict['next_desc_t1'] = None

                    batch.append(row_dict)
            if len(batch) == 0:
                break
            yield collate_fn(batch)

        # batch = []
        # for item in self._fetch_next():
        #     batch.append(item)
        #     if len(batch) == self.loader_size:
        #         yield collate_fn(batch)
        #         batch = []
        # if len(batch):
        #     yield collate_fn(batch)
    def apply_response(self, batch):
        failed_num = 0
        for ptr, response, extract_match, reward_model,extra_info in zip(batch.non_tensor_batch['ptr'], batch.batch['responses'], batch.non_tensor_batch['extract_match'], batch.non_tensor_batch['reward_model'], batch.non_tensor_batch['extra_info']):
            response_text = self.msg_man.tokenizer.decode(response)            
            self.current_response[ptr] = response_text
            if not extract_match:
                failed_num += 1
                if self.patch_threshold > self.error_num[ptr] or self.patch_threshold == -1:
                    step = {}
                    step['action_content'] = reward_model['ground_truth']['check_options']
                    keys_to_remove = ['bbox', 'candidate_bbox','annotation','thought']
                    for key in keys_to_remove:
                        step['action_content'].pop(key, None)
                    print("reward_model['ground_truth']",reward_model['ground_truth'])
                    step['thought'] = reward_model['ground_truth']['thought']
                    ground_truth_response = self.fm.format_response(step,extra_info) # resize coordinate
                    # ground_truth = reward_model['ground_truth']['check_options']
                    self.current_response[ptr] = ground_truth_response
                    self.error_num[ptr] += 1
                else:
                    self.finished[ptr] = True
                    
        return failed_num
            
            
                


def fix_line(line):
    for step in line['steps']:
        check_options = copy.deepcopy(step['action_content'])
        if 'bbox' in step:
            check_options['candidate_bbox'] = step['bbox']
        else:
            check_options['candidate_bbox'] = []
        step['check_options'] = check_options
    return line

if __name__ == "__main__":
    from x.io import read_json
    lines = read_json("androidcontrol_sft_fc_open.std.omniparser.jsonl")
    batch_lines = lines[:16]

    msg_man = QwenMessages2Inputs(
        hf_tokenizer("/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct"),
        {},
        hf_processor("/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct")
    )

    batch_dict = collate_fn([
        {'line': np.array(fix_line(line), dtype=object)}
        for line in batch_lines])
    batch = DataProto.from_single_dict(batch_dict)
    mr_gen = MultiRoundGenerator(batch, rollout_n=5, msg_man=msg_man)
    for sub_batch in mr_gen.fetch_batch():
        print(sub_batch)
        sub_batch = DataProto.from_single_dict(sub_batch)
        for ptr in sub_batch.non_tensor_batch['ptr']:
            mr_gen.current_response[ptr] = '<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"click\", \"coordinate\": [670, 2060]}}\n</tool_call>'
            ## calculate reward
