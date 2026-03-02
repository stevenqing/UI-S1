# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
GUI MultiTurn SFT Dataset for UI-S1 Android control tasks.

This dataset extends MultiTurnSFTDataset to handle messages stored as JSON strings
in parquet files, which is necessary for compatibility with complex nested structures.
"""

import json
import logging
from typing import List, Union

import pandas as pd

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

logger = logging.getLogger(__name__)


def convert_nested_value_to_list_recursive(data_item):
    """Recursively convert nested values to lists."""
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, list):  # Check for numpy array
        import numpy as np
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        return data_item


class GUIMultiTurnSFTDataset(MultiTurnSFTDataset):
    """
    MultiTurn SFT Dataset for GUI trajectory data.

    This dataset handles messages stored as JSON strings in parquet files.
    It deserializes the JSON strings back to Python objects before processing.
    """

    def _read_files_and_process(self):
        """Read parquet files and deserialize JSON messages."""
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # Extract and deserialize messages from JSON strings
        raw_messages = self.dataframe[self.messages_key].tolist()

        self.messages = []
        for msg_str in raw_messages:
            if isinstance(msg_str, str):
                # Deserialize JSON string
                try:
                    self.messages.append(json.loads(msg_str))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse messages JSON: {e}")
                    self.messages.append([])
            elif isinstance(msg_str, list):
                # Already a list, use as-is
                self.messages.append(msg_str)
            else:
                logger.warning(f"Unexpected message type: {type(msg_str)}")
                self.messages.append([])

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None

        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

        logger.info(f"Loaded {len(self.messages)} trajectories")
        logger.info(f"Tools: {'enabled' if self.tools else 'disabled'}")
        logger.info(f"Enable thinking: {'enabled' if self.enable_thinking else 'disabled'}")
