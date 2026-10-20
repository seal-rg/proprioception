import sys
from pathlib import Path

from proprioception import config, distributed, launch
from proprioception.datagen import build
from proprioception.families import resolve_family


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "launch":
        sys.exit(launch.main(argv[1:]))
    stage, args = config.from_argv(argv)
    if stage == "convert":
        from proprioception.primerl_backend import convert_export

        assert isinstance(args, config.ConvertConfig)
        written = convert_export(args.checkpoint)
        print(f"wrote {written}" if written else f"{args.checkpoint} needs no conversion")
        return
    rank, world, device = distributed.setup()
    if world > 1 and isinstance(args, (config.TrainConfig, config.EvalConfig)):
        args.device = device
    if stage == "build":
        assert isinstance(args, config.BuildConfig)
        build(
            args.num_train,
            args.num_eval,
            args.base_url,
            args.seed,
            args.layer,
            Path(args.data_dir),
            resolve_family(args.model),
            args.concurrency,
            args.domain,
            args.lens,
            args.types,
            args.bank_eval_prompts,
            args.lens_token_prob_depth,
            args.answer_opening,
        )
    elif stage == "on-policy-sft":
        from proprioception.on_policy_sft import train_on_policy_sft

        train_on_policy_sft(args)
    elif stage == "eval":
        from proprioception.evaluate import evaluate

        evaluate(args)
    elif stage == "capabilities":
        from proprioception.capabilities import run

        assert isinstance(args, config.CapabilitiesConfig)
        run(args)
    else:
        from proprioception.sft import train_sft

        train_sft(args)
    distributed.teardown()


if __name__ == "__main__":
    main()
