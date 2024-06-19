import os
import warnings

import wandb
from dataloader import load

# import hydra
from hydra.experimental import compose, initialize
from modeling_discriminator import ElectraDiscriminatorWithGenerator
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    ElectraConfig,
    ElectraForMaskedLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

warnings.filterwarnings(action="ignore")


def main(cfg):
    if cfg.ETC.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = cfg.ETC.wandb_project

    # tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL.tokenizer)
    generator = ElectraForMaskedLM.from_pretrained(cfg.MODEL.generator)
    model = ElectraDiscriminatorWithGenerator(
        generator=generator,
        config=ElectraConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            **cfg.MODEL.config,
        ),
        share_embedding_weigh=True,
    )

    # dataset
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    # trining args
    args = TrainingArguments(**cfg.ARGS.training_arguments)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, **cfg.DATA.collator)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 학습 시작
    last_checkpoint = get_last_checkpoint(cfg.ARGS.training_arguments.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(cfg.model.output_dir)
    # 헤드 제거, electra 모델만 저장
    model.electra.save_pretrained(cfg.model.output_dir + "_base")
    tokenizer.save_pretrained(cfg.model.output_dir + "_base")

    if cfg.ETC.get("wandb_project"):
        wandb.finish()


if __name__ == "__main__":
    with initialize(version_base="1.2", config_path="../config/"):
        cfg = compose(config_name="discriminator_config")
    main(cfg)
