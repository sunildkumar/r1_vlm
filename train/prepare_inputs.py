from qwen_vl_utils import process_vision_info


def tokenize_and_inject_images(*, inputs, processing_class):
    """
    inputs: a list of inputs, in this case a list of examples from our dataset
    processing_class: the processing class to use to process the inputs. This is a VLM processor object from the transformers library.

    This is sorta like a collation function but for GRPO

    returns: batch: a batch of inputs ready for training, prompts: The prompts (in text) that were used to generate the batch
    """

    conversations = [ex["messages"] for ex in inputs]

    # Clean up None values from the messages
    for conv in conversations:
        for message in conv:
            content = message["content"]
            message["content"] = [
                {k: v for k, v in item.items() if v is not None} for item in content
            ]
        

    # apply the chat template to the messages and add image tokens
    texts = processing_class.apply_chat_template(
        conversations, continue_final_message=True, tokenize=False
    )

    image_inputs = []
    for conv in conversations:
        image_input, _ = process_vision_info(conv)
        image_inputs.append(image_input)
        

    batch = processing_class(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    # NOTE: There is no need to create labels as we are not autoregressively training!

    return batch, texts



def test_tokenizing(conversations):
    '''
    Debugging function to determine the number of image tokens for different resolutions
    
    # min pixels -> num_image_tokens
    # 256*28*28 -> 256
    # 2*256*28*28 -> 529
    # 224*224 -> 64
    # 2048*2048 -> 5476
    # 1024*28*28 -> 1024
    '''
    from transformers import AutoProcessor
    
    
    
    for pixels in [256*28*28, 2*256*28*28, 224*224, 2048*2048, 1024*28*28]:
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left", min_pixels=pixels, max_pixels=pixels)
        
        texts = processor.apply_chat_template(conversations, continue_final_message=True, tokenize=False)
        
        image_inputs = []   
        for conv in conversations:
            image_input, _ = process_vision_info(conv)
            image_inputs.append(image_input)

        

        batch = processor(text=texts,images=image_inputs,padding=True,return_tensors="pt")

        input_ids = batch["input_ids"]
        num_image_tokens = (input_ids == 151655).sum().item()
        print(f"pixels: {pixels}, num_image_tokens: {num_image_tokens}")
    print('done with test_tokenizing')

    

