import torch
import torch.nn as nn
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining


class ElectraDiscriminatorWithGenerator(ElectraForPreTraining):
    def __init__(
        self,
        generator: ElectraForMaskedLM,
        config: ElectraConfig,
        share_embedding_weigh: bool = True,
    ):
        super.__init__(config)

        self.generator = generator
        # weight freeze
        for _, child in self.generator.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def weight_sharing(self):
        state_dict = self.generator.electra.embeddings.state_dict()
        self.electra.embeddings.load_state_dict(state_dict)

        for _, child in self.electra.embeddings.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):

        generator_hidden_states = self.generator(input_ids=input_ids, attention_mask=attention_mask)
        generator_output = torch.argmax(generator_hidden_states.logits, dim=-1)
        idx = labels > -100

        corrupted_input_ids = input_ids.clone()
        corrupted_input_ids[idx] = generator_output[idx]
        discriminator_labels = torch.where(idx, 1, labels.new_zeros(labels.size()))

        return super().__call__(
            input_ids=corrupted_input_ids,
            attention_mask=attention_mask,
            labels=discriminator_labels,
        )
