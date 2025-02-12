import argparse

from transformers import LlamaForCausalLM, LlamaTokenizer

from tran_qwen_to_llama import tran_qwen_to_llama


def main():
    parser = argparse.ArgumentParser(description="A script to use model_llama.py to activate different model")
    parser.add_argument("model_name", type=str, help="To activate different kinds of model")
    
    args = parser.parse_args()
    
    if args.model_name == "qwen":
        model_path = tran_qwen_to_llama(model = "Qwen", path = "./downloads/Qwen2-0.5B", target = "./verify_version")
    else:
        print("Can't find the correct model")
        return
    
    
    
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    
    
    prompt = "Who are you?"
    inputs = tokenizer(prompt, return_tensors = "pt")
    generate_ids = model.generate(inputs.input_ids, max_new_tokens = 50)
    output = tokenizer.decode(generate_ids[0], skip_special_tokens = True)
    
    print(output)
    


if __name__ == "__main__":
    main()