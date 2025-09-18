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
Preprocess the limo.jsonl dataset to parquet format
"""

import re
import os
import json
import random
import argparse
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from datasets import Dataset


def make_prefix(question, template_type):
    """Create a formatted prompt with instructions for the model."""
    if template_type == 'base':
        """This works for any base model"""
        prefix = f'Answer the given math question. Please reason step by step and put the final answer in \\boxed{{}}. Question: {question} '
    else:
        raise NotImplementedError

    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/limo_data')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--input_file', default='./data/limo.jsonl', help='Path to the input JSONL file')
    parser.add_argument('--train_test_split', type=float, default=0.1, help='Portion of data to use for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    data_source = 'limo'

    # Create output directory
    os.makedirs(args.local_dir, exist_ok=True)

    # Load data from JSONL file
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} examples from {args.input_file}")
    
    # Shuffle the data with a fixed seed for reproducibility
    random.seed(args.seed)
    random.shuffle(data)
    
    # Split into train/test
    split_index = int(len(data) * (1 - args.train_test_split))
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    print(f"Split into {len(train_data)} training examples and {len(test_data)} test examples")

    # Create datasets
    def process_data(examples, split_name):
        processed_data = []
        
        for idx, item in enumerate(examples):
            # Extract fields from the data
            question = item.get('question', '').strip()
            solution = item.get('solution', '')
            answer = item.get('answer', '')
            
            # Ensure question ends with a proper punctuation
            if question and question[-1] not in ['?', '.']:
                question += '?'
                
            # Format the question with the template
            formatted_question = make_prefix(question, template_type=args.template_type)
            
            # Create the processed data item
            processed_item = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": formatted_question,
                }],
                "ability": "math-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": [answer],  # Wrap answer in a list to match NQ format
                    }
                },
                "extra_info": {
                    'split': split_name,
                    'index': idx,
                    'solution': solution  # Store the solution for reference
                }
            }
            
            processed_data.append(processed_item)
        
        return processed_data

    # Process train and test data
    train_processed = process_data(train_data, 'train')
    test_processed = process_data(test_data, 'test')
    
    # Convert to Datasets
    train_dataset = Dataset.from_list(train_processed)
    test_dataset = Dataset.from_list(test_processed)
    
    # Save to parquet files
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    
    print(f"Saved processed datasets to {args.local_dir}/train.parquet and {args.local_dir}/test.parquet")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        print(f"Copying to HDFS directory: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"Copy to HDFS complete")