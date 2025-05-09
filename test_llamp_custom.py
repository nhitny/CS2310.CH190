import torch

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import os
import json
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

import deepspeed
from flags import parser, DATA_FOLDER
from data.meta_dataset import MetaDataset
from models.common import Classification
from models.llamp import LLaMP

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.integrations import HfDeepSpeedConfig
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
path_reults = os.getenv("PATH_RESULTS")

def main():
    local_parser = deepspeed.add_config_arguments(parser)
    local_parser.add_argument("--target_dataset", type=str)
    args = local_parser.parse_args()
    logpath = args.logpath
    dataset = args.dataset

    os.makedirs(logpath, exist_ok=True)

    with open(args.deepspeed_config, "r") as fp:
        deepspeed_config = json.load(fp)
    dschf = HfDeepSpeedConfig(deepspeed_config)

    llama_model = LlamaForCausalLM.from_pretrained(
        args.model_base, device_map="cpu", token=hf_token
    )
    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_base, device_map="cpu", token=hf_token
    )

    base_testset = MetaDataset(
        phase="val", dataset=dataset, num_shots=args.coop_num_shots, seed=args.coop_seed
    )
    new_testset = MetaDataset(
        phase="test", dataset=dataset, num_shots=args.coop_num_shots, seed=args.coop_seed
    )
    classnames = {"base": base_testset.classnames, "new": new_testset.classnames}

    base_loader = torch.utils.data.DataLoader(
        base_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers
    )
    new_loader = torch.utils.data.DataLoader(
        new_testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers
    )

    evaluator_base = Classification(args, base_testset.idx2label)
    evaluator_new = Classification(args, new_testset.idx2label)
    device = torch.device(args.device)

    if os.path.isdir(args.load):
        ckpt_dir = args.load
    else:
        ckpt_dir = os.path.dirname(args.load)

    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".t7")])
    print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")

    results = []

    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        print(f"Loading checkpoint from: {ckpt_path}")

        model = LLaMP(base_testset, classnames, args, llama_model, tokenizer, few_shot=False)

        try:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Failed to load model from checkpoint {ckpt_path}: {e}")
            continue

        model_engine, _, _, _ = deepspeed.initialize(config=deepspeed_config, model=model)
        model_engine.eval()

        with torch.no_grad():
            base_acc = test(0, model_engine, base_loader, evaluator_base, args, logpath, device, subset="Base")["accuracy"]
            new_acc = test(0, model_engine, new_loader, evaluator_new, args, logpath, device, subset="New")["accuracy"]
            hm = 2 * base_acc * new_acc / (base_acc + new_acc)

            print(f"{ckpt_file}: Base: {base_acc:.4f}, New: {new_acc:.4f}, HM: {hm:.4f}")

            results.append({
                "dataset": dataset,
                "checkpoint": ckpt_file,
                "Base": round(base_acc, 4),
                "New": round(new_acc, 4),
                "HM": round(hm, 4),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    if results:
        base_avg = np.mean([r["Base"] for r in results])
        new_avg = np.mean([r["New"] for r in results])
        hm_avg = np.mean([r["HM"] for r in results])
        results.append({
            "dataset": dataset,
            "checkpoint": "AVERAGE",
            "Base": round(base_avg, 4),
            "New": round(new_avg, 4),
            "HM": round(hm_avg, 4),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(path_reults, f"results_{dataset.lower()}.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved all results to {results_csv_path}")

def test(epoch, model, testloader, evaluator, args, logpath, device, subset):
    evaluator.reset()
    model.eval()
    model.module.compute_all_class_embeddings(subset=subset.lower())

    for _, data in tqdm(
        enumerate(testloader), total=len(testloader), desc=f"Testing on {subset}"
    ):
        data = [d.to(device) for d in data]
        data[0] = data[0].bfloat16()
        data[1] = data[1].bfloat16()

        with torch.inference_mode():
            _, predictions = model(data, subset=subset.lower())

        predictions = predictions.cpu()
        evaluator.process(predictions, data[-1].cpu())

    stats = evaluator.evaluate()
    stats["a_epoch"] = epoch

    summary = " | ".join([f"{k}: {round(v, 4)}" for k, v in stats.items()])
    print(f"Test Epoch {epoch} [{subset}]: {summary}")
    return stats

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
