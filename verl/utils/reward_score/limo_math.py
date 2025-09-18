import sys
import os
import random
import re

from verl.utils.reward_score.limo_utils.grader import math_equal, check_is_correct
from verl.utils.reward_score.limo_utils.parser import extract_answer, strip_string
from verl.utils.reward_score.limo_utils.math_normalization import normalize_final_answer

# def extract_number_from_answer(answer_text):
#     """
#     Extracts a number from the answer text, handling various formats.
    
#     Args:
#         answer_text: The text from which to extract numbers
        
#     Returns:
#         str: The extracted number or original text if no clear number is found
#     """
#     # If the answer is already purely numeric, return it
#     if answer_text.isdigit() or (answer_text.replace('.', '', 1).isdigit() and answer_text.count('.') == 1):
#         return answer_text
    
#     # Check for LaTeX boxed content
#     boxed_matches = re.findall(r'\\boxed\{(.*?)\}', answer_text)
#     if boxed_matches:
#         # Return the content of the boxed environment
#         return boxed_matches[0].strip()
    
#     # Strip common prefixes and suffixes that often surround the actual answer
#     prefixes = ["the answer is ", "answer: ", "is ", "equals ", "= ", "is equal to "]
#     for prefix in prefixes:
#         if answer_text.lower().startswith(prefix):
#             answer_text = answer_text[len(prefix):]
    
#     # Try to find numbers in the text
#     # First, look for integers and decimals
#     number_matches = re.findall(r'[-+]?\d+\.?\d*', answer_text)
#     if number_matches:
#         # If the answer is embedded in sentences like "The 100th term is 981",
#         # we want to prefer the last number as it's typically the actual answer
#         if len(number_matches) > 1:
#             # Check for common patterns indicating the last number is the answer
#             last_part = answer_text.split()[-3:]  # Last few words
#             last_part_text = ' '.join(last_part).lower()
#             if any(term in last_part_text for term in ["is", "equals", "="]):
#                 return number_matches[-1]
        
#         # Otherwise return the number that looks most like a final answer
#         # Prefer longer numbers or the last one
#         return max(number_matches, key=len) if number_matches else number_matches[-1]
    
#     # Try to find fractions like "3/4"
#     fraction_matches = re.findall(r'[-+]?\d+\s*/\s*\d+', answer_text)
#     if fraction_matches:
#         return fraction_matches[0].replace(" ", "")
    
#     # Try to find fractions written with \frac{}{} notation
#     frac_matches = re.findall(r'\\frac\{(\d+)\}\{(\d+)\}', answer_text)
#     if frac_matches:
#         num, denom = frac_matches[0]
#         return f"{num}/{denom}"
    
#     # If no clear number is found, return the original text
#     return answer_text

def extract_number_from_answer(answer_text):
    """
    Extracts a number from the answer text, handling various formats, including nested braces in \\boxed.

    Args:
        answer_text: The text from which to extract numbers

    Returns:
        str: The extracted number or original text if no clear number is found
    """
    # If the answer is already purely numeric, return it
    if answer_text.isdigit() or (answer_text.replace('.', '', 1).isdigit() and answer_text.count('.') == 1):
        return answer_text

    # Check for LaTeX boxed content
    boxed_start_indices = [match.start() for match in re.finditer(r'\\boxed{', answer_text)]  # Find ALL starts

    if boxed_start_indices:
        last_boxed_start_index = boxed_start_indices[-1] # Take the LAST one.
        try:
            if answer_text[last_boxed_start_index + len('\\boxed')] == '{':
                content_start_char_index = last_boxed_start_index + len('\\boxed{')
                open_braces = 1
                content_end_char_index = -1
                for i in range(content_start_char_index, len(answer_text)):
                    char = answer_text[i]
                    if char == '{':
                        open_braces += 1
                    elif char == '}':
                        open_braces -= 1
                        if open_braces == 0:
                            content_end_char_index = i
                            break
                if content_end_char_index != -1:
                    return answer_text[content_start_char_index:content_end_char_index].strip()
        except IndexError:
            pass  # Handle potential index errors

    # Strip common prefixes and suffixes
    prefixes = ["the answer is ", "answer: ", "is ", "equals ", "= ", "is equal to "]
    for prefix in prefixes:
        if answer_text.lower().startswith(prefix):
            answer_text = answer_text[len(prefix):]

    # Try to find numbers in the text
    # First, look for integers and decimals
    number_matches = re.findall(r'[-+]?\d+\.?\d*', answer_text)
    if number_matches:
        if len(number_matches) > 1:
            last_part = answer_text.split()[-3:]
            last_part_text = ' '.join(last_part).lower()
            if any(term in last_part_text for term in ["is", "equals", "="]):
                return number_matches[-1]
        return max(number_matches, key=len) if number_matches else number_matches[-1]

    # Try to find fractions like "3/4"
    fraction_matches = re.findall(r'[-+]?\d+\s*/\s*\d+', answer_text)
    if fraction_matches:
        return fraction_matches[0].replace(" ", "")

    # Try to find fractions written with \frac{}{} notation
    frac_matches = re.findall(r'\\frac\{(\d+)\}\{(\d+)\}', answer_text)
    if frac_matches:
        num, denom = frac_matches[0]
        return f"{num}/{denom}"

    # If no clear number is found, return the original text
    return answer_text

def clean_system_guidance(solution_str):
    """
    Remove system-generated guidance text and system content from the solution string.
    
    Args:
        solution_str: The solution text with potential system guidance and system content
        
    Returns:
        str: Clean solution without system guidance and system content
    """
    
    new_guidance_pattern = r"My previous action is invalid\. \nIf I want to give the final answer, I should put the answer between <answer> and </answer>\. Let me try again:"
    # new_guidance_pattern = r"My previous response probably failed to wrap the final answer in the required tags, or exceeded the token limit\. \nIf I want to give the final answer, I should put the answer between <answer> and </answer>\. Continue my previous response:"
    # 使用 re.sub 将匹配到的模式替换为空字符串，从而实现删除
    # flags=re.DOTALL 在这里不是必需的，因为模式中的 \n 显式处理了换行，
    # 并且模式中没有使用 . 来跨越多行。但如果其他地方有此需求，可以加上。
    clean_solution = re.sub(new_guidance_pattern, '', solution_str)
    
#     # More comprehensive pattern to match the full guidance message
    guidance_pattern = r'My previous action is invalid If I want to give the final answer[^:]*?Let me try again:'
    
    # Split by the guidance pattern and join the parts
    parts = re.split(guidance_pattern, solution_str, flags=re.DOTALL)
    clean_solution = ''.join(parts)
    
    # Also handle any standalone guidance messages without the "Continue my previous response:" part
#     My previous action is invalid. \
# If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again:
    guidance_standalone = r'My previous action is invalid If I want to give the final answer[^:]*?If I want to (ask|give|keep)'
    parts = re.split(guidance_standalone, clean_solution, flags=re.DOTALL)
    clean_solution = ''.join(parts)
    
    # Remove system content in <communicate> tags - handle potentially nested tags
    # We'll use a while loop to repeatedly clean until no more communicate tags are found
    prev_solution = ""
    while prev_solution != clean_solution:
        prev_solution = clean_solution
        communicate_pattern = r'<communicate>(.*?)</communicate>'
        parts = re.split(communicate_pattern, clean_solution, flags=re.DOTALL)
        
        # Join parts, but replace the system content parts with an empty string
        cleaned_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Even indices are outside <communicate> tags
                cleaned_parts.append(part)
            # Odd indices are inside <communicate> tags, we skip them
        
        clean_solution = ''.join(cleaned_parts)
    
    # Remove all ds Response lines and any text coming after until a newline
    # ds_response_pattern = r'ds Response.*?(?:\n|$)'
    # clean_solution = re.sub(ds_response_pattern, '', clean_solution, flags=re.MULTILINE)
    
#             chat_data_modified.insert(0, {
#             'role': 'system',
#             'content':  """The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
# The answer is enclosed within <answer> </answer> tags. i.e., <answer> answer here </answer>
# During the assistant's reasoning process, if he realizes that his reasoning may be problematic or wrong, he can ask other agents for help. 
# The query is inclosed within <ask> </ask> Tags. i.e., <ask> put confused point here </ask>. It will return the advice from other agent within <communicate> </communicate>. 
# The assistant can ask other agents for help multiple times.
# If the assistant understand the question and find no further other agents' advice needed, the assistant can directly provide the answer inside <answer> </answer>"""
#         })
# The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
# The answer is enclosed within <answer> </answer> tags. i.e., <answer> answer here </answer>.
# If the assistant understand the question and find no need for further reasoning, the assistant can directly provide the answer inside <answer> </answer>."""

    # Remove system-instruction content that might have been copied into the solution
    system_instruction_patterns = [
        r'The user asks a question, and the Assistant solves it.*?(?:\n\n|$)',
        r'The answer is enclosed within <answer> </answer> tags.*?(?:\n\n|$)',
        r'If the assistant understand the question.*?(?:\n\n|$)'
    ]
    
    for pattern in system_instruction_patterns:
        clean_solution = re.sub(pattern, '', clean_solution, flags=re.DOTALL)
    
    return clean_solution

def compute_score_limo_r1(solution_str, ground_truth, format_score=0.2, score=1.0, return_components=False):
    """
    LIMO-based math scoring function.
    
    Args:
        solution_str: The solution text
        ground_truth: The ground truth
        format_score: Score to give for partial correctness
        score: Score for fully correct answer
        return_components: Whether to return individual components of the score
    """
    
    # Random logging for debugging - increase chance for debugging the issue
    do_print = True  
    if do_print:
        print(solution_str)
    # First clean the solution from system guidance and system content
    clean_solution = clean_system_guidance(solution_str)
    
    answer_text = "" # Initialize
    extracted_using_boxed = False

    # Priority 1: Find the last \boxed{} and extract its full content
    # Find all starting positions of \boxed{
    boxed_start_indices = [match.start() for match in re.finditer(r'\\boxed{', clean_solution)]
    
    if boxed_start_indices:
        last_boxed_start_index = boxed_start_indices[-1]
        
        # Robustly find the content within the braces for the last \boxed{}
        # Start searching from the '{' immediately after '\\boxed'
        try:
            if clean_solution[last_boxed_start_index + len('\\boxed')] == '{':
                content_start_char_index = last_boxed_start_index + len('\\boxed{')
                open_braces = 1 # Count for the initial opening brace
                content_end_char_index = -1
                for i in range(content_start_char_index, len(clean_solution)):
                    char = clean_solution[i]
                    if char == '{':
                        # Only increment if it's not the very first char we are inspecting (which made open_braces=1)
                        # This logic is slightly off; the first char IS the start of content.
                        # Corrected: if open_braces was already 1 due to the char at content_start_char_index-1
                        # then char at content_start_char_index could be another '{'.
                        # Simpler: initialize open_braces = 1 for the first '{' at content_start_char_index -1
                        # then loop from content_start_char_index.
                        open_braces += 1
                    elif char == '}':
                        open_braces -= 1
                        if open_braces == 0: # This is the key: when the initial brace is balanced
                            content_end_char_index = i
                            break
                
                if content_end_char_index != -1:
                    answer_text = clean_solution[content_start_char_index:content_end_char_index].strip()
                    extracted_using_boxed = True
            # else: \boxed was not followed by { - malformed, ignore.
        except IndexError:
            pass # \boxed{ at the very end of string, malformed.

    if not extracted_using_boxed:
        # Priority 2: Find the last <answer> </answer> Tag
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_tag_matches = list(re.finditer(answer_pattern, clean_solution, re.DOTALL))
        if answer_tag_matches:
            answer_text_from_tag = answer_tag_matches[-1].group(1).strip()
            # For <answer>, extract numbers
            answer_text = extract_number_from_answer(answer_text_from_tag)
        else:
            # Priority 3: Find keywords like "Final answer:", "Answer:"
            # Find all keyword matches and take the last one's content
            last_keyword_match_content = None
            keyword_patterns = [
                r'(?i)(?:Final\sAnswer|Answer)\s*:\s*(.+?)(?:\n\n|$)', # Non-greedy, stop at double newline or end
                r'(?i)(?:The\sFinal\sAnswer\sis|The\sAnswer\sis)\s*(.+?)(?:\n\n|$)'
            ]
            
            all_keyword_matches = []
            for pattern in keyword_patterns:
                for match in re.finditer(pattern, clean_solution, re.DOTALL):
                    all_keyword_matches.append(match)
            
            if all_keyword_matches:
                all_keyword_matches.sort(key=lambda m: m.start())
                last_match = all_keyword_matches[-1]
                answer_text_from_keyword = last_match.group(1).strip()
                answer_text = extract_number_from_answer(answer_text_from_keyword)
            else:
                # Priority 4 (Fallback): Original LIMO's extraction if nothing else found
                answer_text = extract_answer(clean_solution) # from limo_utils.parser
    
    # Calculate format score based on is_R1 parameter - using the clean solution
    # format_reward = calculate_format_score_r1(clean_solution)
    format_reward = 0.0
    
    # if no boxed packaged answer, return format reward only
    if not answer_text: # This checks if answer_text is empty after all attempts
        # if do_print:
        #     print("--------------------------------")
        #     print(f"Golden answer: {ground_truth}")
        #     print(f"Extracted answer: None")
        #     print(f"Format reward: {format_reward}")
        # accuracy_reward = 0.0
        if return_components:
            accuracy_reward = 0.0
            return format_reward, 0.0, format_reward + accuracy_reward
        return format_reward
        
    # Clean and prepare answer
    answer_text_processed = strip_string(answer_text)
    
    extracted_number_for_comparison = extract_number_from_answer(answer_text_processed)
    
    # Normalize the answer for better matching
    normalized_answer_full = normalize_final_answer(answer_text_processed)
    normalized_extracted_number = normalize_final_answer(extracted_number_for_comparison)
    
    # Use LIMO's math_equal for sophisticated checking
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        gt_answer = ground_truth['target']
    else:
        gt_answer = ground_truth
    
    # Debug printing
    if do_print:
        print("--------------------------------")
        print(f"Golden answer type: {type(gt_answer)}")
        if isinstance(gt_answer, list):
            print(f"Golden answer: {gt_answer} (list with {len(gt_answer)} elements)")
            # for i, item in enumerate(gt_answer):
            #     print(f"  Item {i}: {item} (type: {type(item)})")
        else:
            print(f"Golden answer: {gt_answer}")
        print(f"Raw answer_text from extraction: '{answer_text}' (used for answer_text_processed)")
        print(f"Processed answer_text_processed: '{answer_text_processed}' (used for normalized_answer_full)")
        print(f"Extracted_number_for_comparison: '{extracted_number_for_comparison}' (used for normalized_extracted_number)")
        print(f"Normalized_answer_full: '{normalized_answer_full}'")
        print(f"Normalized_extracted_number: '{normalized_extracted_number}'")
        print(f"Format reward: {format_reward}")
        print(f"Extracted using boxed: {extracted_using_boxed}")
    
    accuracy_reward = 0.0
    
    # Check if the normalized answer is a number
    is_normalized_full_numeric = normalized_answer_full.isdigit() or \
                               (normalized_answer_full.replace('.', '', 1).isdigit() and normalized_answer_full.count('.') <= 1)
    is_normalized_extracted_numeric = normalized_extracted_number.isdigit() or \
                                    (normalized_extracted_number.replace('.', '', 1).isdigit() and normalized_extracted_number.count('.') <= 1)
    
    primary_candidate_for_math_equal = normalized_answer_full
    secondary_candidate_for_math_equal = normalized_extracted_number

    # Support multiple formats for ground truth
    if isinstance(gt_answer, list):
        for i, gt in enumerate(gt_answer):
            # Convert gt to string for comparison
            gt_str = str(gt).strip() if not isinstance(gt, str) else gt.strip()
            
            # Try math_equal with the primary candidate
            try:
                if math_equal(primary_candidate_for_math_equal, gt_str):
                    if do_print:
                        print(f"CORRECT! (math_equal with primary: '{primary_candidate_for_math_equal}') Item {i}")
                    accuracy_reward = score
                    break
            except Exception as e:
                if do_print:
                    print(f"Error in math_equal with primary for item {i}: {e}")
            
            # If primary is complex and secondary is simpler and different, try math_equal with secondary
            if accuracy_reward == 0 and primary_candidate_for_math_equal != secondary_candidate_for_math_equal:
                try:
                    if math_equal(secondary_candidate_for_math_equal, gt_str):
                        if do_print:
                            print(f"CORRECT! (math_equal with secondary: '{secondary_candidate_for_math_equal}') Item {i}")
                        accuracy_reward = score
                        break
                except Exception as e:
                    if do_print:
                        print(f"Error in math_equal with secondary for item {i}: {e}")
            
            # Fallback to check_is_correct with original answer_text_processed (before number extraction)
            if accuracy_reward == 0:
                try:
                    if check_is_correct(answer_text_processed, gt_str): # uses original string before num extraction
                        if do_print:
                            print(f"CORRECT! (check_is_correct with answer_text_processed: '{answer_text_processed}') Item {i}")
                        accuracy_reward = score
                        break
                except Exception as e:
                    if do_print:
                        print(f"Error in check_is_correct (answer_text_processed) for item {i}: {e}")
            
            # Fallback to check_is_correct with extracted_number_for_comparison
            if accuracy_reward == 0 and answer_text_processed != extracted_number_for_comparison: # only if different
                try:
                    if check_is_correct(extracted_number_for_comparison, gt_str):
                        if do_print:
                            print(f"CORRECT! (check_is_correct with extracted_number_for_comparison: '{extracted_number_for_comparison}') Item {i}")
                        accuracy_reward = score
                        break
                except Exception as e:
                    if do_print:
                        print(f"Error in check_is_correct (extracted_number_for_comparison) for item {i}: {e}")
                    
            if do_print and accuracy_reward == 0:
                print(f"Item {i}: Neither '{primary_candidate_for_math_equal}' nor '{secondary_candidate_for_math_equal}' matched '{gt_str}'")
                
        if do_print and accuracy_reward == 0:
            print(f"INCORRECT. No matches found in list of {len(gt_answer)} items.")
                
    else:
        # Single ground truth
        gt_str = str(gt_answer).strip() if not isinstance(gt_answer, str) else gt_answer.strip()
        
        # Try math_equal with the primary candidate
        try:
            if math_equal(primary_candidate_for_math_equal, gt_str):
                if do_print:
                    print(f"CORRECT! (math_equal with primary: '{primary_candidate_for_math_equal}')")
                accuracy_reward = score
        except Exception as e:
            if do_print:
                print(f"Error in math_equal with primary: {e}")
                
        # If primary is complex and secondary is simpler and different, try math_equal with secondary
        if accuracy_reward == 0 and primary_candidate_for_math_equal != secondary_candidate_for_math_equal:
            try:
                if math_equal(secondary_candidate_for_math_equal, gt_str):
                    if do_print:
                        print(f"CORRECT! (math_equal with secondary: '{secondary_candidate_for_math_equal}')")
                    accuracy_reward = score
            except Exception as e:
                if do_print:
                    print(f"Error in math_equal with secondary: {e}")
                
        # Fallback to check_is_correct with original answer_text_processed
        if accuracy_reward == 0:
            try:
                if check_is_correct(answer_text_processed, gt_str): # uses original string before num extraction
                    if do_print:
                        print(f"CORRECT! (check_is_correct with answer_text_processed: '{answer_text_processed}')")
                    accuracy_reward = score
            except Exception as e:
                if do_print:
                    print(f"Error in check_is_correct (answer_text_processed): {e}")
        
        # Fallback to check_is_correct with extracted_number_for_comparison
        if accuracy_reward == 0 and answer_text_processed != extracted_number_for_comparison: # only if different
            try:
                if check_is_correct(extracted_number_for_comparison, gt_str):
                    if do_print:
                        print(f"CORRECT! (check_is_correct with extracted_number_for_comparison: '{extracted_number_for_comparison}')")
                    accuracy_reward = score
            except Exception as e:
                if do_print:
                    print(f"Error in check_is_correct (extracted_number_for_comparison): {e}")
            
        if do_print and accuracy_reward == 0:
            print(f"INCORRECT. Neither '{primary_candidate_for_math_equal}' nor '{secondary_candidate_for_math_equal}' matched '{gt_str}'")
    
    if return_components:
        return format_reward, accuracy_reward, format_reward + accuracy_reward
    return format_reward + accuracy_reward

def compute_score_limo_base(solution_str, ground_truth, format_score=0.2, score=1.0, return_components=False):
    """
    LIMO-based math scoring function.
    
    Args:
        solution_str: The solution text
        ground_truth: The ground truth
        format_score: Score to give for partial correctness
        score: Score for fully correct answer
        return_components: Whether to return individual components of the score
    """
    do_print = random.randint(1, 2) == 1  
    if do_print:
        print(solution_str)
    # First clean the solution from system guidance and system content
    
    clean_solution = clean_system_guidance(solution_str)
    
    # Extract answer from the <answer> tags first, using the clean solution
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, clean_solution, re.DOTALL))
    
    # Random logging for debugging - increase chance for debugging the issue
    
    if len(matches) >= 1:
        # Take the last match if multiple exist
        answer_text = matches[-1].group(1).strip()
    else:
        # Check for \boxed{} content directly if <answer> tags not found
        boxed_pattern = r'\\boxed\{(.*?)\}'
        boxed_matches = list(re.finditer(boxed_pattern, clean_solution, re.DOTALL))
        if boxed_matches:
            answer_text = boxed_matches[-1].group(1).strip()
        else:
            # Fall back to LIMO's extraction if no tags or boxed content found
            answer_text = extract_answer(clean_solution)
    
    # Calculate format score - using the clean solution
    # format_reward = calculate_format_score_r1(clean_solution)
    format_reward = 0.0
    
    # if no boxed packaged answer, return format reward only
    if not answer_text:
        if do_print:
            print("--------------------------------")
            print(f"Golden answer: {ground_truth}")
            print(f"Extracted answer: None")
            print(f"Format reward: {format_reward}")
        accuracy_reward = 0.0
        if return_components:
            accuracy_reward=0.0
            return format_reward, accuracy_reward, format_reward + accuracy_reward
        return format_reward
        
    # Clean and prepare answer
    answer_text = strip_string(answer_text)
    
    # Try to extract a numerical answer if present
    extracted_number = extract_number_from_answer(answer_text)
    
    # Normalize the answer for better matching
    normalized_answer = normalize_final_answer(answer_text)
    normalized_number = normalize_final_answer(extracted_number)
    
    # Use LIMO's math_equal for sophisticated checking
    if isinstance(ground_truth, dict) and 'target' in ground_truth:
        gt_answer = ground_truth['target']
    else:
        gt_answer = ground_truth
    
    # Debug printing
    if do_print:
        print("--------------------------------")
        # print(f"Golden answer type: {type(gt_answer)}")
        # if isinstance(gt_answer, list):
        #     print(f"Golden answer: {gt_answer} (list with {len(gt_answer)} elements)")
        #     for i, item in enumerate(gt_answer):
        #         print(f"  Item {i}: {item} (type: {type(item)})")
        # else:
        #     print(f"Golden answer: {gt_answer}")
        print(f"Extracted answer: {answer_text}")
        print(f"Extracted number: {extracted_number}")
        # print(f"Normalized answer: {normalized_answer} (type: {type(normalized_answer)})")
        # print(f"Normalized number: {normalized_number} (type: {type(normalized_number)})")
        print(f"Format reward: {format_reward}")
    
    accuracy_reward = 0.0
    
    # Check if the normalized answer is a number
    is_normalized_numeric = normalized_answer.isdigit() or (normalized_answer.replace('.', '', 1).isdigit() and normalized_answer.count('.') <= 1)
    is_extracted_numeric = normalized_number.isdigit() or (normalized_number.replace('.', '', 1).isdigit() and normalized_number.count('.') <= 1)
    
    # Support multiple formats for ground truth
    if isinstance(gt_answer, list):
        for i, gt in enumerate(gt_answer):
            # Convert gt to string for comparison
            gt_str = str(gt).strip() if not isinstance(gt, str) else gt.strip()
            
            # First try direct comparison with the extracted number
            if is_extracted_numeric and gt_str.isdigit() and normalized_number == gt_str:
                if do_print:
                    print(f"CORRECT! (extracted number match) Item {i}: '{normalized_number}' == '{gt_str}'")
                accuracy_reward = score
                break
            
            # Try math_equal with the extracted number
            try:
                if math_equal(normalized_number, gt_str):
                    if do_print:
                        print(f"CORRECT! (math_equal with extracted number) Item {i}: math_equal({normalized_number}, {gt_str}) is True")
                    accuracy_reward = score
                    break
            except Exception as e:
                if do_print:
                    print(f"Error in math_equal with extracted number for item {i}: {e}")
            
            # Fall back to original logic with normalized_answer
            # Direct numerical comparison with original normalized answer
            if is_normalized_numeric and gt_str.isdigit() and normalized_answer == gt_str:
                if do_print:
                    print(f"CORRECT! (numeric string match) Item {i}: '{normalized_answer}' == '{gt_str}'")
                accuracy_reward = score
                break
            
            # Try math_equal with original normalized answer
            try:
                if math_equal(normalized_answer, gt_str):
                    if do_print:
                        print(f"CORRECT! (math_equal) Item {i}: math_equal({normalized_answer}, {gt_str}) is True")
                    accuracy_reward = score
                    break
            except Exception as e:
                if do_print:
                    print(f"Error in math_equal for item {i}: {e}")
            
            # Fallback to check_is_correct with both extracted number and original answer
            try:
                if check_is_correct(extracted_number, gt_str):
                    if do_print:
                        print(f"CORRECT! (check_is_correct with extracted number) Item {i}: check_is_correct({extracted_number}, {gt_str}) is True")
                    accuracy_reward = score
                    break
            except Exception as e:
                if do_print:
                    print(f"Error in check_is_correct with extracted number for item {i}: {e}")
                    
            try:
                if check_is_correct(answer_text, gt_str):
                    if do_print:
                        print(f"CORRECT! (check_is_correct) Item {i}: check_is_correct({answer_text}, {gt_str}) is True")
                    accuracy_reward = score
                    break
            except Exception as e:
                if do_print:
                    print(f"Error in check_is_correct for item {i}: {e}")
                    
            if do_print:
                print(f"Item {i}: Neither '{normalized_answer}' nor '{normalized_number}' matched '{gt_str}'")
                
        if do_print and accuracy_reward == 0:
            print(f"INCORRECT. No matches found in list of {len(gt_answer)} items.")
                
    else:
        # Single ground truth
        gt_str = str(gt_answer).strip() if not isinstance(gt_answer, str) else gt_answer.strip()
        
        # First try direct comparison with the extracted number
        if is_extracted_numeric and gt_str.isdigit() and normalized_number == gt_str:
            if do_print:
                print(f"CORRECT! (extracted number match): '{normalized_number}' == '{gt_str}'")
            accuracy_reward = score
        else:
            # Try math_equal with the extracted number
            try:
                if math_equal(normalized_number, gt_str):
                    if do_print:
                        print(f"CORRECT! (math_equal with extracted number): math_equal({normalized_number}, {gt_str}) is True")
                    accuracy_reward = score
                else:
                    if do_print:
                        print(f"math_equal({normalized_number}, {gt_str}) is False")
            except Exception as e:
                if do_print:
                    print(f"Error in math_equal with extracted number: {e}")
                
            # If still no match, try with the original normalized answer
            if accuracy_reward == 0:
                # Direct numerical comparison
                if is_normalized_numeric and gt_str.isdigit() and normalized_answer == gt_str:
                    if do_print:
                        print(f"CORRECT! (numeric string match): '{normalized_answer}' == '{gt_str}'")
                    accuracy_reward = score
                else:
                    # Try math_equal
                    try:
                        if math_equal(normalized_answer, gt_str):
                            if do_print:
                                print(f"CORRECT! (math_equal): math_equal({normalized_answer}, {gt_str}) is True")
                            accuracy_reward = score
                        else:
                            if do_print:
                                print(f"math_equal({normalized_answer}, {gt_str}) is False")
                    except Exception as e:
                        if do_print:
                            print(f"Error in math_equal: {e}")
                
            # If still no match, try check_is_correct with both extracted number and original answer
            if accuracy_reward == 0:
                try:
                    if check_is_correct(extracted_number, gt_str):
                        if do_print:
                            print(f"CORRECT! (check_is_correct with extracted number): check_is_correct({extracted_number}, {gt_str}) is True")
                        accuracy_reward = score
                    else:
                        if do_print:
                            print(f"check_is_correct({extracted_number}, {gt_str}) is False")
                except Exception as e:
                    if do_print:
                        print(f"Error in check_is_correct with extracted number: {e}")
                        
                if accuracy_reward == 0:
                    try:
                        if check_is_correct(answer_text, gt_str):
                            if do_print:
                                print(f"CORRECT! (check_is_correct): check_is_correct({answer_text}, {gt_str}) is True")
                            accuracy_reward = score
                        else:
                            if do_print:
                                print(f"check_is_correct({answer_text}, {gt_str}) is False")
                    except Exception as e:
                        if do_print:
                            print(f"Error in check_is_correct: {e}")
            
        if do_print and accuracy_reward == 0:
            print(f"INCORRECT. Neither '{normalized_answer}' nor '{normalized_number}' matched '{gt_str}'")
    
    if return_components:
        accuracy_reward=0.0
        return format_reward, accuracy_reward, format_reward + accuracy_reward
    return format_reward + accuracy_reward

def calculate_format_score_base(solution_str, is_R1=False):
    """
    Calculate format score based on the presence of specific tags,
    with system content removed from consideration.
    
    Args:
        solution_str: The solution text
        is_R1: Whether to use R1-specific format scoring
    
    Returns:
        float: Format score between 0 and 1
    """
    # Filter out system-generated guidance text and system content
    clean_solution = clean_system_guidance(solution_str)
    
    # Debug print for checking the cleaning effectiveness
    # if random.randint(1, 100) == 1:  # 1% chance to print
    #     print("--- Format scoring Base ---")
    #     print(f"Original solution length: {len(solution_str)}")
    #     print(f"Clean solution length: {len(clean_solution)}")
    #     print(f"Original solution start: {solution_str[:200]}...")
    #     print(f"Clean solution start: {clean_solution[:200]}...")
    #     if len(solution_str) != len(clean_solution):
    #         print(f"Removed system content: {len(solution_str) - len(clean_solution)} characters")
    
    # Non-R1 format scoring: Check for <think>, <answer>, and <ask> tags
    # Each tag present adds weighted score
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'
    ask_pattern = r'<ask>(.*?)</ask>'
    
    think_matches = list(re.finditer(think_pattern, clean_solution, re.DOTALL))
    answer_matches = list(re.finditer(answer_pattern, clean_solution, re.DOTALL))
    ask_matches = list(re.finditer(ask_pattern, clean_solution, re.DOTALL))
    
    format_score = 0.0
    
    # Add score for each tag
    think_score = 0.1 if len(think_matches) >= 1 else 0.0
    answer_score = 0.1 if len(answer_matches) >= 1 else 0.0
    ask_score = 0.1 if len(ask_matches) >= 1 else 0.0
    
    format_score = think_score + answer_score + ask_score
    
    # Apply penalty if no expected formatting is found
    if format_score == 0.0:
        format_score = -0.05
        
    # Additional debug logging
    # if random.randint(1, 100) == 1:  # 1% chance to print
    #     print(f"  <think> tags found: {len(think_matches)} (score: {think_score})")
    #     print(f"  <answer> tags found: {len(answer_matches)} (score: {answer_score})")
    #     print(f"  <ask> tags found: {len(ask_matches)} (score: {ask_score})")
    #     print(f"  Total format score: {format_score}")
    
    # Clamp to [0, 1] range
    return max(0.0, min(1.0, format_score))

def calculate_format_score_r1(solution_str, is_R1=False):
    """
    Calculate format score based on the presence of specific tags,
    with system content removed from consideration.
    
    Args:
        solution_str: The solution text
        is_R1: Whether to use R1-specific format scoring
    
    Returns:
        float: Format score between 0 and 1
    """
    # Filter out system-generated guidance text and system content
    clean_solution = clean_system_guidance(solution_str)
    
    # Debug print for checking the cleaning effectiveness
    # if random.randint(1, 100) == 1:  # 1% chance to print
    #     print("--- Format scoring R1 ---")
    #     print(f"Original solution length: {len(solution_str)}")
    #     print(f"Clean solution length: {len(clean_solution)}")
    #     print(f"Original solution start: {solution_str[:200]}...")
    #     print(f"Clean solution start: {clean_solution[:200]}...")
    #     if len(solution_str) != len(clean_solution):
    #         print(f"Removed system content: {len(solution_str) - len(clean_solution)} characters")
    
    # R1 format scoring: <ask> tag presence gives format_score=1, otherwise 0
    ask_pattern = r'<ask>(.*?)</ask>'
    ask_matches = list(re.finditer(ask_pattern, clean_solution, re.DOTALL))
    
    # Additional debug logging
    # if random.randint(1, 100) == 1:  # 1% chance to print
    #     print(f"  <ask> tags found: {len(ask_matches)}")
    #     print(f"  Format score: {1.0 if len(ask_matches) >= 1 else 0.0}")
    
    return 0.5 if len(ask_matches) >= 1 else 0.0

def is_system_guidance(text):
    """
    Check if the text matches the system-generated guidance pattern.
    
    Args:
        text: The text to check
        
    Returns:
        bool: True if the text appears to be system guidance
    """
    # More comprehensive pattern to detect guidance text
    guidance_patterns = [
        r'My previous response probably failed to wrap the final answer/query in the required tags',
        r'If I want to ask other agents for help, I should put the query between <ask> and </ask>',
        r'If I want to give the final answer, I should put the answer between <answer> and </answer>',
        r'If I want to keep thinking, I should put the thinking between <think> and </think>',
        r'Continue my previous response:'
    ]
    
    # Check if any of the patterns match
    for pattern in guidance_patterns:
        if re.search(pattern, text, re.DOTALL):
            return True
    return False

def is_system_content(text):
    """
    Check if the text appears to be system-generated content.
    
    Args:
        text: The text to check
        
    Returns:
        bool: True if the text appears to be system content
    """
    # Patterns to detect system content
    system_patterns = [
        r'<communicate>.*?</communicate>',        # Content between communicate tags
        r'ds Response',                           # ds Response lines
        r'The user asks a question, and the Assistant solves it',  # System instruction
        r'During the assistant\'s reasoning process',              # System instruction
        r'The query is inclosed within',                           # System instruction
        r'The assistant can ask other agents',                     # System instruction
        r'If the assistant understand the question',               # System instruction
    ]
    
    # Check if any of the patterns match
    for pattern in system_patterns:
        if re.search(pattern, text, re.DOTALL):
            return True
    return False