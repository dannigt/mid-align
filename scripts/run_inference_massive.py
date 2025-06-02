from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from peft import PeftModel, PeftConfig
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
import json
from pathlib import Path

from .utils import convert_row_massive_slot_filling as convert_row_func
from .utils import massive_lang_map


def main(args):
    result_directory = Path(args.result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(args.base_model_name,
                                                 offload_folder="offload", offload_state_dict=True,
                                                 torch_dtype=torch.bfloat16,
                                                 ).to(args.device)
    if args.peft_model_id is None:
        print("*** Running base model without any finetuned component!")
    else:
        peft_config = PeftConfig.from_pretrained(args.peft_model_id)
        model = PeftModel.from_pretrained(model, args.peft_model_id,
                                          torch_dtype=torch.float16,
                                          config=peft_config)

    generation_kwargs = {"max_new_tokens": args.max_new_tokens,
                         "num_beams": args.beam_size,
                         "do_sample": False,
                         }
    dataset = load_dataset("AmazonScience/massive", name=massive_lang_map[args.lang])

    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load data
    partition = args.partition
    dataset[partition] = dataset[partition].map(convert_row_func)
    print(dataset[partition])

    print("\n*** First 5 samples:")
    for i in range(5):
        print(f"input: {dataset[partition][i]['utt']}\n"
              f"output: {dataset[partition][i]['annot_utt']}")

    system_prompt = {
        "role": "system",
        "content": "Given a command from the user, a voice assistant will extract entities essential for carry out the command.\n"
                   "For example, for a command about playing a specific song, the name of the song mentioned by the user would be an entity, falling under the type of \"song name\".\n"
                   "Your task is to extract the entities as words from the command if they fall under a predefined list of entity types."
    }

    datalist = []
    for sample in dataset[partition]:
        template = [system_prompt, {"role": "user", "content": sample["utt"]}]
        prefix_in_template = tokenizer.apply_chat_template(template,
                                                           tokenize=False,
                                                           add_generation_prompt=True)
        # To be compatible with prompt template used in our pretrained llama model
        if args.no_default_template and args.base_model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
            prefix_in_template = prefix_in_template.replace("<|eot_id|>", "<|eot_id|>\n").replace("<|begin_of_text|>",
                                                                                                  "")
        datalist.append(prefix_in_template)

    print("\n1st sample:")
    print(datalist[0])

    dataloader = DataLoader(datalist, batch_size=args.bsz)

    # start generating
    all_prompt, all_generated = [], []
    for entry in tqdm(dataloader):
        with torch.no_grad():
            tokenized_data = tokenizer(entry, padding=True, return_tensors='pt').to(args.device)
            prompt_len = int(tokenized_data.attention_mask.shape[1])
            outputs = model.generate(input_ids=tokenized_data.input_ids,
                                     attention_mask=tokenized_data.attention_mask,
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id,
                                     **generation_kwargs)

            generated = tokenizer.batch_decode([output[prompt_len:] for output in outputs],
                                               skip_special_tokens=True)
            all_prompt.extend(entry)
            all_generated.extend(generated)

    with open(result_directory / args.output_path, "w") as f:
        for prompt_line, output_line in zip(all_prompt, all_generated):
            f.write(json.dumps({"prompt": prompt_line,
                                "prediction": output_line},
                               ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model-name", type=str,
                        default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='model name in the hub or local path')
    parser.add_argument("--peft-model-id", type=str,
                        help="path to peft model")
    parser.add_argument("--lang", type=str, default='en', help="language used in inference")
    parser.add_argument("--device", type=str, default='cuda:0', help="language used in inference")
    parser.add_argument("--partition", type=str, default="validation")
    parser.add_argument("--output-path", type=str, default="output_inference_massive.json")
    parser.add_argument("--result-directory", type=str, default="./")
    parser.add_argument("--bsz", type=int, default=64, help="batch size")
    parser.add_argument("--beam-size", type=int, default=1, help="beam size")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="max number of generated new tokens")
    parser.add_argument("--no-default-template", action="store_true")
    args = parser.parse_args()

    main(args)
