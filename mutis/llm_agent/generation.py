import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig

from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
from typing import List
import time
import concurrent.futures
import threading
import traceback

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    # logging: dict
    num_gpus: int
    no_think_rl: bool=False

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        # logger: Tracking,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Save original responses before processing
        original_responses_str = responses_str.copy()
        
        # Process each response
        processed_responses = []
        for resp in responses_str:
            # First check for standard <answer> tags
            if '</answer>' in resp:
                processed_resp = resp.split('</answer>')[0] + '</answer>'
            else:
                # Check for "Answer:" or "Final Answer:" format
                answer_match = re.search(r'(?:Answer|Final Answer)\s*:', resp)
                if answer_match:
                    # Find the position of the match
                    start_pos = answer_match.start()
                    # Extract everything up to the end of the next paragraph or end of text
                    end_pos = resp[start_pos:].find('\n\n')
                    if end_pos == -1:  # No paragraph break found
                        processed_resp = resp  # Keep the entire response
                    else:
                        processed_resp = resp[:start_pos + end_pos]
                # Check for \boxed{} LaTeX format
                elif '\\boxed{' in resp:
                    # Find the position of the first \boxed
                    start_pos = resp.find('\\boxed{')
                    # Find the corresponding closing brace
                    brace_count = 0
                    end_pos = -1
                    for i in range(start_pos + 7, len(resp)):  # +7 to skip '\boxed{'
                        if resp[i] == '{':
                            brace_count += 1
                        elif resp[i] == '}':
                            if brace_count == 0:
                                end_pos = i
                                break
                            brace_count -= 1
                    
                    if end_pos != -1:
                        processed_resp = resp[:end_pos + 1]  # Include the closing brace
                    else:
                        processed_resp = resp  # Keep entire response if no closing brace found
                else:
                    processed_resp = resp  # No markers found, keep as is
            
            processed_responses.append(processed_resp)

        responses_str = processed_responses

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str, original_responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        import time
        import threading
        import traceback
        
        # Set a global timeout for the entire method
        global_timeout_seconds = 3600  # 1 hour max for the entire loop
        start_time = time.time()
        
        # Sets up original_left_side to hold the initial prompt
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        # Creates empty original_right_side to accumulate responses and masked info
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        # tracking tensors
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Initialize conversation history for each active example
        conversation_history = ["" for _ in range(active_mask.shape[0])]
        
        # Main generation loop
        for step in range(self.config.max_turns):
            # Check global timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > global_timeout_seconds:
                print(f"CRITICAL WARNING: run_llm_loop global timeout reached after {elapsed_time:.2f} seconds! Stopping loop.")
                break
                
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # Set a timeout for the model generation step
            generation_timeout = 600  # 10 minutes max for generation
            generation_result = None
            generation_exception = None
            generation_completed = threading.Event()
            
            def run_model_generation():
                nonlocal generation_result, generation_exception
                try:
                    # Note: creating a new DataProto for active samples to avoid modifying the original
                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })            
                    gen_output = self._generate_with_gpu_padding(rollings_active)
                    generation_result = gen_output
                except Exception as e:
                    generation_exception = e
                    print(f"Exception in model generation: {e}")
                    traceback.print_exc()
                finally:
                    generation_completed.set()
            
            # Run generation in a separate thread
            generation_thread = threading.Thread(target=run_model_generation)
            generation_thread.daemon = True
            generation_thread.start()
            
            # Wait for completion with timeout
            generation_success = generation_completed.wait(timeout=generation_timeout)
            
            if not generation_success:
                print(f"CRITICAL WARNING: Model generation timed out after {generation_timeout} seconds! Stopping loop.")
                break
                
            if generation_exception:
                print(f"Model generation failed with exception: {generation_exception}")
                break
                
            if generation_result is None:
                print("No generation result produced. Stopping loop.")
                break
                
            gen_output = generation_result
            meta_info = gen_output.meta_info
            
            # Set a timeout for post-processing
            post_processing_timeout = 60  # 1 minute for post-processing
            post_processing_completed = threading.Event()
            post_processing_result = None
            post_processing_exception = None
            
            def run_post_processing():
                nonlocal post_processing_result, post_processing_exception
                try:
                    responses_ids, responses_str, original_responses_str = self._postprocess_responses(gen_output.batch['responses'])
                    responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
                    post_processing_result = (responses_ids, responses_str, original_responses_str)
                except Exception as e:
                    post_processing_exception = e
                    print(f"Exception in post-processing: {e}")
                    traceback.print_exc()
                finally:
                    post_processing_completed.set()
            
            # Run post-processing in a separate thread
            post_thread = threading.Thread(target=run_post_processing)
            post_thread.daemon = True
            post_thread.start()
            
            # Wait for completion with timeout
            post_success = post_processing_completed.wait(timeout=post_processing_timeout)
            
            if not post_success:
                print(f"CRITICAL WARNING: Post-processing timed out after {post_processing_timeout} seconds! Stopping loop.")
                break
                
            if post_processing_exception:
                print(f"Post-processing failed with exception: {post_processing_exception}")
                break
                
            if post_processing_result is None:
                print("No post-processing result produced. Stopping loop.")
                break
                
            responses_ids, responses_str, original_responses_str = post_processing_result

            # Update conversation history for active examples
            active_idx = 0
            for i in range(len(conversation_history)):
                if active_mask[i]:
                    if conversation_history[i]:
                        conversation_history[i] += "\n\n" + original_responses_str[active_idx]
                    else:
                        conversation_history[i] = original_responses_str[active_idx]
                    active_idx += 1

            # Execute in environment and process observations
            try:
                # Set a timeout for execution
                execution_timeout = 300  # 5 minutes for execution
                execution_start = time.time()
                
                next_obs, dones, valid_action = self.execute_predictions_with_agent(
                    responses_str, self.tokenizer.pad_token, active_mask, 
                    original_responses=original_responses_str,
                    conversation_history=conversation_history
                )
                
                execution_duration = time.time() - execution_start
                if execution_duration > 60:  # Log if it took more than a minute
                    print(f"Agent execution took {execution_duration:.2f} seconds")
                    
            except Exception as e:
                print(f"CRITICAL ERROR in execute_predictions_with_agent: {e}")
                traceback.print_exc()
                # Create fallback responses to allow the loop to continue
                next_obs = []
                dones = []
                valid_action = []
                
                for a in active_mask:
                    if a:
                        next_obs.append('\n\n<communicate>Error in processing. Please provide your final answer.</communicate>\n\n')
                        dones.append(0)  # Don't mark as done to allow another attempt
                        valid_action.append(0)
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            # 根据生成的结果进行更新
            try:
                rollings = self._update_rolling_state(
                    rollings,
                    responses_ids,
                    next_obs_ids
                )
                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                    next_obs_ids
                )
            except Exception as e:
                print(f"Error updating state: {e}")
                traceback.print_exc()
                # Try to continue without updating if possible
                print("Attempting to continue despite state update error")
            
        # final LLM rollout
        if active_mask.sum():
            # Check global timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > global_timeout_seconds:
                print(f"CRITICAL WARNING: run_llm_loop global timeout reached before final rollout, after {elapsed_time:.2f} seconds!")
            else:
                try:
                    rollings.batch = self.tensor_fn.cut_to_effective_len(
                        rollings.batch,
                        keys=['input_ids', 'attention_mask', 'position_ids']
                    )

                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })            
                    
                    # Set a timeout for the final generation
                    final_generation_timeout = 600  # 10 minutes max
                    final_generation_completed = threading.Event()
                    final_generation_result = None
                    final_generation_exception = None
                    
                    def run_final_generation():
                        nonlocal final_generation_result, final_generation_exception
                        try:
                            gen_output = self._generate_with_gpu_padding(rollings_active)
                            final_generation_result = gen_output
                        except Exception as e:
                            final_generation_exception = e
                            print(f"Exception in final generation: {e}")
                            traceback.print_exc()
                        finally:
                            final_generation_completed.set()
                    
                    # Run final generation in a separate thread
                    final_thread = threading.Thread(target=run_final_generation)
                    final_thread.daemon = True
                    final_thread.start()
                    
                    # Wait for completion with timeout
                    final_success = final_generation_completed.wait(timeout=final_generation_timeout)
                    
                    if not final_success:
                        print(f"CRITICAL WARNING: Final model generation timed out after {final_generation_timeout} seconds!")
                    elif final_generation_exception:
                        print(f"Final model generation failed with exception: {final_generation_exception}")
                    elif final_generation_result is not None:
                        gen_output = final_generation_result
                        meta_info = gen_output.meta_info
                        
                        # Process final responses
                        try:
                            responses_ids, responses_str, original_responses_str = self._postprocess_responses(gen_output.batch['responses'])
                            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

                            # Update conversation history for final responses
                            active_idx = 0
                            for i in range(len(conversation_history)):
                                if active_mask[i]:
                                    if conversation_history[i]:
                                        conversation_history[i] += "\n\n" + original_responses_str[active_idx]
                                    else:
                                        conversation_history[i] = original_responses_str[active_idx]
                                    active_idx += 1

                            # Execute final predictions
                            _, dones, valid_action = self.execute_predictions_with_agent(
                                responses_str, self.tokenizer.pad_token, active_mask, 
                                original_responses=original_responses_str,
                                conversation_history=conversation_history
                            )

                            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                            active_mask = active_mask * curr_active_mask
                            active_num_list.append(active_mask.sum().item())
                            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
                            
                            original_right_side = self._update_right_side(
                                original_right_side,
                                responses_ids,
                            )
                        except Exception as e:
                            print(f"Error in final response processing: {e}")
                            traceback.print_exc()
                except Exception as e:
                    print(f"Error in final LLM rollout: {e}")
                    traceback.print_exc()
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        
        # Record total execution time
        total_elapsed = time.time() - start_time
        meta_info['total_execution_time'] = total_elapsed
        print(f"ACTIVE_TRAJ_NUM: {active_num_list}")
        print(f"Total execution time: {total_elapsed:.2f} seconds")
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions_with_agent(self, predictions: List[str], pad_token: str, active_mask=None, original_responses=None, conversation_history=None) -> List[str]:
        """
        Execute predictions and return observations.
        
        Args:
            predictions: List of action predictions
            pad_token: Token to use for padding
            active_mask: Mask of active examples
            original_responses: Original unprocessed responses containing full thinking context
            conversation_history: Complete history of the conversation for each example
            
        Returns:
            Tuple of (next_obs, dones, valid_action)
        """
        # Set a global timeout for the entire method
        global_timeout = 300  # 5 minutes in seconds
        
        # Create a threading event to signal completion
        completion_event = threading.Event()
        
        # Results containers that can be accessed from the thread
        result_next_obs = []
        result_dones = []
        result_valid_action = []
        
        def execute_with_timeout():
            try:
                cur_actions, contents, thinking_contexts = self.postprocess_predictions(predictions)
                next_obs, dones, valid_action = [], [], []

                # Process results for each item in batch
                for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
                    if not active:
                        next_obs.append('')
                        dones.append(1)
                        valid_action.append(0)
                    else:
                        if action == 'answer':
                            next_obs.append('')
                            dones.append(1)
                            valid_action.append(1)
                        else: # Handles invalid actions
                            next_obs.append(f'\n My previous action is invalid. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again:\n')
                            dones.append(0)
                            valid_action.append(0)
                
                # Assign results to the outer scope containers
                result_next_obs.extend(next_obs)
                result_dones.extend(dones)
                result_valid_action.extend(valid_action)
                
                # Set completion event
                completion_event.set()
                
            except Exception as e:
                print(f"Critical error in execute_predictions_with_agent: {e}")
                # Fill with default values in case of exception
                for i, active in enumerate(active_mask):
                    if active:
                        result_next_obs.append(f'\n\n<communicate>Global timeout occurred. Please provide your final answer based on what you know.</communicate>\n\n')
                        result_dones.append(0)
                        result_valid_action.append(0)
                    else:
                        result_next_obs.append('')
                        result_dones.append(1)
                        result_valid_action.append(0)
                # Set completion event
                completion_event.set()
        
        # Start execution in a separate thread
        execution_thread = threading.Thread(target=execute_with_timeout)
        execution_thread.daemon = True
        execution_thread.start()
        
        # Wait for completion with timeout
        completion_success = completion_event.wait(timeout=global_timeout)
        
        if not completion_success:
            print(f"CRITICAL WARNING: execute_predictions_with_agent timed out after {global_timeout} seconds!")
            # Fill any remaining slots with default values
            if not result_next_obs:
                for i, active in enumerate(active_mask):
                    if active:
                        result_next_obs.append(f'\n\n<communicate>Global timeout occurred. Please provide your final answer based on what you know.</communicate>\n\n')
                        result_dones.append(0)
                        result_valid_action.append(0)
                    else:
                        result_next_obs.append('')
                        result_dones.append(1)
                        result_valid_action.append(0)
        
        return result_next_obs, result_dones, result_valid_action

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[str], List[str]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, content list, full thinking context list)
        """
        actions = []
        contents = []
        thinking_contexts = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                # First check for standard <answer> tags. <ask> tag processing removed.
                tag_pattern = r'<(answer)>(.*?)</\1>' # Only 'answer'
                tag_match = re.search(tag_pattern, prediction, re.DOTALL)
                
                # Use full prediction as thinking context
                thinking_contexts.append(prediction)
                
                if tag_match:
                    content = tag_match.group(2).strip()  # Return only the content inside the tags
                    action = tag_match.group(1) # Will be 'answer'
                else:
                    # Check for additional answer markers if no tags were found
                    # 1. Check for "Answer:" format
                    answer_prefix_pattern = r'(?i)(?:Answer|Final Answer)\s*:(.*?)(?=$|\n\n)'
                    answer_prefix_match = re.search(answer_prefix_pattern, prediction, re.DOTALL)
                    
                    # 2. Check for \boxed{} LaTeX format
                    boxed_pattern = r'\\boxed\{(.*?)\}'
                    boxed_match = re.search(boxed_pattern, prediction, re.DOTALL)
                    
                    if answer_prefix_match:
                        content = answer_prefix_match.group(1).strip()
                        # print("answer string matched")
                        action = 'answer'  # Treat as answer action
                    elif boxed_match:
                        content = boxed_match.group(1).strip()
                        action = 'answer'  # Treat as answer action
                        # print("box string matched")
                    else:
                        content = ''
                        action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents, thinking_contexts

