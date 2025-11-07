from scripts.dataset import VQAv2Dataset
from scripts.evaluate import Evaluator
from scripts.metric import ExactMatch, VQASoftAcc, DelayStats
from scripts.model import LlavaVisionZipModel

# ============ #
# Entry (CLI)  #
# ============ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--ann_path", type=str, required=True, help="VQAv2-like annotation json")
    parser.add_argument("--output_dir", type=str, default="./outputs_vqav2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dominant", type=int, default=54)
    parser.add_argument("--contextual", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="vqa", choices=["vqa", "mme", 'pope'],
                    help="datatset options")
    args = parser.parse_args()

    model = LlavaVisionZipModel(
        model_path=args.model_path,
        dominant=args.dominant,
        contextual=args.contextual,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    # dataset = VQAv2Dataset(args.ann_path, limit=args.limit)
    if args.dataset == "vqa":
        from scripts.dataset import VQAv2Dataset
        from scripts.metric import ExactMatch, VQASoftAcc
        dataset = VQAv2Dataset(args.ann_path, limit=args.limit)
        metrics = [ExactMatch(), VQASoftAcc(), DelayStats()]
    elif args.dataset == "mme":
        from scripts.dataset import MMEDataset
        from scripts.metric import MMEAcc, MMEAccPlus
        dataset = MMEDataset(args.ann_path, limit=args.limit)
        metrics = [MMEAcc(), MMEAccPlus(), DelayStats()]
    elif args.dataset == "pope":
        from scripts.dataset import POPEDataset
        from scripts.metric import POPEAcc, POPEPrecisionRecallF1
        dataset = POPEDataset(args.ann_path, limit=args.limit)
        metrics = [POPEAcc(), POPEPrecisionRecallF1(), DelayStats()]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    evaluator = Evaluator(model, dataset, metrics, output_dir=args.output_dir, warmup=args.warmup, seed=args.seed)
    evaluator.run(limit=args.limit)