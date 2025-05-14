from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from peft import PeftModel, PeftConfig
import evaluate
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
import json
from pathlib import Path

from .utils import convert_row_translation as convert_row_func
from .utils import translation_lang_map


def main(args):
    result_directory = Path(args.result_directory)
    result_directory.mkdir(parents=True, exist_ok=True)

    translation_dir = f"{args.source_lang}-{args.lang}"
    if not translation_dir == "cs-en":  # load the reverse version (error in dataset)
        dataset = load_dataset("haoranxu/WMT23-Test", name=translation_dir)
        dataset = dataset.rename_column(translation_dir, "translation")
    else:
        dataset = load_dataset("haoranxu/WMT23-Test", name=f"{args.lang}-{args.source_lang}")
        dataset = dataset.rename_column(f"{args.lang}-{args.source_lang}", "translation")

    # Load data
    partition = "test"
    dataset[partition] = dataset[partition].map(convert_row_func,
                                                fn_kwargs={"sl": args.source_lang, "tl": args.lang})
    print(dataset[partition])

    if not args.evaluation_only:
        model = AutoModelForCausalLM.from_pretrained(args.base_model_name,
                                                     offload_folder="offload",
                                                     offload_state_dict=True,
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

        # prepare tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        # print examples
        print("\n*** First 5 samples:")
        for i in range(5):
            print(f"input: {dataset[partition][i]['src']}\n"
                  f"output: {dataset[partition][i]['tgt']}")

        system_prompt = {
            "role": "system",
            "content": f"Translate the following sentences "
                       f"from {translation_lang_map[args.source_lang]} "
                       f"to {translation_lang_map[args.lang]}."
        }

        datalist = []
        for sample in dataset[partition]:
            template = [system_prompt, {"role": "user", "content": sample["src"]}]
            prefix_in_template = tokenizer.apply_chat_template(template,
                                                               tokenize=False,
                                                               add_generation_prompt=True)
            # To be compatible with prompt template used in our pretrained llama model
            if not args.use_default_template and args.base_model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                prefix_in_template.replace("<|eot_id|>", "<|eot_id|>\n").replace("<|begin_of_text|>", "")
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

        with open(args.output_path, "w") as f:
            for prompt_line, output_line in zip(all_prompt, all_generated):
                f.write(json.dumps({"prompt": prompt_line,
                                    "prediction": output_line},
                                   ensure_ascii=False) + '\n')
    else:
        result_lines = []
        with open(args.output_path, "r") as f:
            for line in f:
                result_lines.append(json.loads(line))
        all_generated = [line['prediction'] for line in result_lines]

    # evaluation
    sources, references = dataset[partition]["src"], dataset[partition]["tgt"]
    sacrebleu, comet = evaluate.load("sacrebleu"), evaluate.load("comet")
    sacrebleu_results = sacrebleu.compute(predictions=all_generated,
                                          references=[[ref] for ref in references])
    comet_results = comet.compute(sources=sources,
                                  references=references,
                                  predictions=all_generated)
    del comet_results["scores"]     # remove individual scores to avoid clutter
    with open(result_directory / f"{args.source_lang}-{args.lang}_bleu.txt", "w") as f:
        f.write(json.dumps(sacrebleu_results))
    with open(result_directory / f"{args.source_lang}-{args.lang}_comet.txt", "w") as f:
        f.write(json.dumps(comet_results))
    print("BLEU:", sacrebleu_results)
    print("COMET:", comet_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model-name", type=str,
                        default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='model name in the hub or local path')
    parser.add_argument("--peft-model-id", type=str,
                        help="path to peft model")
    parser.add_argument("--lang", type=str, default='en', help="language used in inference")
    parser.add_argument("--source-lang", type=str, default='en', help="source language used in inference")
    parser.add_argument("--device", type=str, default='cuda:0', help="language used in inference")
    parser.add_argument("--output-path", type=str, default="output_inference_massive.json")
    parser.add_argument("--result-directory", type=str, default="./")
    parser.add_argument("--bsz", type=int, default=32, help="batch size")
    parser.add_argument("--beam-size", type=int, default=1, help="beam size")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="max number of generated new tokens")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--use-default-template", action="store_false")
    args = parser.parse_args()

    main(args)
