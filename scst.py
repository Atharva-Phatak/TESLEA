import pytorch_lightning as pl
import torch
from utils import save_json, compute_mean
import torch.nn.functional as F
from transformers import (
    MaxLengthCriteria,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)
from filtering import top_k_top_p_filtering, ngram_copy_filtering

# from scoring import calculate_rouge_reward
from hf_data import create_data
import os
from typing import Tuple, Dict
from scoring import combine_rewards
from utils import create_weight_vector
from collections import OrderedDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RLTrainer(pl.LightningModule):
    def __init__(self, model_cfg, node_cfg, generate_cfg, ul_config, reward_list):
        super().__init__()
        self.model_cfg = model_cfg
        self.generate_cfg = generate_cfg
        self.node_cfg = node_cfg
        self.config = AutoConfig.from_pretrained(self.model_cfg.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_cfg.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.model_name)
        self.start = self.model_cfg.start
        self.stop = self.model_cfg.stop
        self.train_ds = create_data(
            model=self.model, split="train", start=self.start, stop=self.stop
        )
        self.top_k = self.generate_cfg.top_k
        self.top_p = self.generate_cfg.top_p
        self.no_copy_ngram = self.generate_cfg.no_copy_ngram
        self.no_repeat_ngram = self.generate_cfg.no_repeat_ngram
        self.val_rouge, self.test_results = None, None
        self.ul_config = ul_config
        self.ul_weights = (
            self.init_ul_weights() if self.ul_config.use_ul is True else None
        )
        self.reward_list = reward_list
        self.are_weights_on_device = False

    def init_ul_weights(self) -> torch.Tensor:
        tokens = set([int(i) for i in self.ul_config.exclude_tokens.split(",")])
        ids, weights = create_weight_vector(
            fname=self.ul_config.ul_file_path,
            num_weights=self.ul_config.ul_n_weights,
            exclude_tokens=tokens,
        )
        if self.ul_config.ul_softmax:
            weights = torch.nn.functional.softmax(
                weights / self.ul_config.ul_temprature, dim=-1
            )
        weight_vector = torch.zeros(self.config.vocab_size).float()
        for i in range(len(ids)):
            weight_vector[ids[i]] = weights[i]

        return weight_vector

    @property
    def decoder_start_id(self) -> int:
        """Property to get decoder start id."""
        return self.model.config.decoder_start_token_id

    @property
    def eos_token_id(self) -> int:
        """property to get eos token id."""
        return self.model.config.eos_token_id

    @property
    def get_encoder(self):
        """Get encoder part of encoder decoder model."""
        return self.model.get_encoder()

    @staticmethod
    def _update_past(past, new_past) -> Dict:

        past["past"] = new_past
        return past

    def unlikelihood_loss(
        self, decoder_input_ids, logits, weight_mask, selective_penalty=False
    ):
        probs = F.softmax(logits, dim=-1)
        neg_probs = 1 - probs

        # replace zeros with small positive constant for stability
        neg_probs += (neg_probs == 0).float() * 1e-8
        log_neg_probs = torch.log(neg_probs)  # (N,s,v)

        # now create attention mask and apply it
        attention_mask = decoder_input_ids.eq(1).eq(0).float()
        attention_mask = attention_mask.unsqueeze(2).expand(-1, -1, logits.shape[2])
        log_neg_probs_masked = log_neg_probs * attention_mask

        # apply weight vector to the log probability tensor
        N, s = logits.size()[:2]
        weight_mask_expanded = weight_mask.unsqueeze(0).unsqueeze(0).expand(N, s, -1)
        weighted_probs = log_neg_probs_masked * weight_mask_expanded

        if selective_penalty:
            indices = torch.argmax(logits, dim=-1)
            indices_mask = F.one_hot(indices, num_classes=logits.shape[-1])  # (N,s,v)
            weighted_probs *= indices_mask

            # now determine the number of tokens to which UL is applied
            count_vec = (weighted_probs != 0).int()  # (N,s,v)
            count_vec = torch.sum(count_vec, dim=-1)  # (N,s)
            pad_mask = decoder_input_ids.eq(1).eq(0).int()
            count_vec *= pad_mask

            # self.num_outputs += pad_mask.sum()
            # self.num_ul += count_vec.sum()

        # TODO: take into account batch size (doesn't matter now since N=1)
        ul = -torch.sum(weighted_probs)
        if ul.isinf().any().item() is True:
            ul = torch.tensor(0.0)
        elif ul.isnan().any().item() is True:
            ul = torch.tensor(0.0)
        # print(weighted_probs)
        # z = torch.count_nonzero(weighted_probs)
        # print(ul)
        # print(ul.isnan().any())
        return ul

    def create_logits_processors(self) -> LogitsProcessorList:
        """Create HF logits processor."""
        wrapper = LogitsProcessorList()
        no_repeat_ngram = (
            self.sample_cfg.no_repeat_ngram_size
            if self.sample_cfg.no_repeat_ngram_size
            else None
        )
        length_penalty = self.sample_cfg.len_pen if self.sample_cfg.len_pen else None
        if no_repeat_ngram is not None:
            wrapper.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram))
        if length_penalty is not None:
            wrapper.append(RepetitionPenaltyLogitsProcessor(length_penalty))
        return wrapper if len(wrapper) > 0 else None

    def create_stopping_criteria(self) -> StoppingCriteriaList:
        """Create hf stopping criteria."""
        assert self.sample_cfg.max_len is not None, "Max length must be specified."

        wrapper = StoppingCriteriaList(
            MaxLengthCriteria(max_length=self.sample_cfg.max_len)
        )
        return wrapper

    def create_sampled_logits_wrapper(self) -> LogitsProcessorList:
        """Create HF sampled logits processors."""
        top_k = self.sample_cfg.top_k
        top_p = self.sample_cfg.top_p
        temprature = self.sample_cfg.temprature
        wrapper = LogitsProcessorList()
        if temprature is not None:
            self.sampled_logits_preprocessor.append(
                TemperatureLogitsWarper(temperature=temprature)
            )
        if top_k is not None:
            self.sampled_logits_preprocessor.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            self.sampled_logits_preprocessor.append(TopPLogitsWarper(top_p=top_p))

        return wrapper if len(wrapper) > 0 else None

    def _expand_inputs_for_generation(self, num_return_sequences):
        # Todo implement beam search
        pass

    def _get_encoder_outputs(self, input_ids, attention_mask) -> Dict:
        """Get encoder outputs."""
        encoder = self.get_encoder
        encoder_outputs = encoder(input_ids, attention_mask, return_dict=True)
        return encoder_outputs

    def _decode(self, decoded, past) -> Tuple:
        """Get decoder input with past."""
        decoder_input = self.model.prepare_inputs_for_generation(
            decoder_input_ids=decoded,
            past=past["past"],
            attention_mask=past["attention_mask"],
            encoder_outputs=past["encoder_outputs"],
        )
        output = self.model(**decoder_input)
        past = self._update_past(past=past, new_past=output.past_key_values)
        return output.logits, past

    def sample_log_probs(
        self, input_ids, attention_mask, max_len, target_mask, is_greedy=False
    ) -> Tuple:
        """Method to sample next sequence tokens using top-p, top-k or greedy"""
        N = input_ids.shape[0]
        log_probs = []
        encoder_outputs = self._get_encoder_outputs(
            input_ids=input_ids, attention_mask=attention_mask
        )
        past = {
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "past": None,
        }
        build_up = (
            torch.tensor([self.model.config.bos_token_id])
            .repeat(N, 1)
            .type_as(input_ids)
        )
        end_id = self.tokenizer.eos_token_id
        finished_func = lambda build_up: all(
            [end_id in build for build in build_up[:, 1:]]
        )

        while build_up.shape[-1] <= max_len and not finished_func(build_up):
            logits, past = self._decode(decoded=build_up, past=past)
            logits = logits[:, -1, :]

            logits = ngram_copy_filtering(
                build_up, input_ids, logits, n_gram=self.no_copy_ngram
            )
            logits = ngram_copy_filtering(
                build_up, build_up, logits, n_gram=self.no_repeat_ngram
            )
            logits = top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p)

            # Handle min_length here
            # if min_length > 0 and build_up.shape[1] <= min_length:

            if is_greedy is False:
                multinomial_dist = torch.distributions.Categorical(logits=logits)
                current = multinomial_dist.sample()
                log_prob = multinomial_dist.log_prob(current)
                log_probs.append(log_prob)
            else:
                log_prob = torch.nn.functional.softmax(logits, dim=-1)
                current = torch.argmax(log_prob, dim=-1)

            current = current.view(-1, 1)
            build_up = torch.cat((build_up, current), dim=1)

        if is_greedy is False:
            log_probs = torch.stack(log_probs, dim=1)
            # log_probs = log_probs * target_mask  # Not considering sampled words with padding mask = 0
            lens = torch.sum(target_mask, dim=1)  # Length of sampled sentence
            log_probs = torch.sum(log_probs, dim=1) / lens  # (bs,)

        return log_probs, build_up

    def forward(self, batch) -> torch.Tensor:
        """Forward pass for the models."""

        outputs = self.model(
            input_ids=batch["input_ids"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
        )
        # print(outputs.loss)
        if self.ul_config.use_ul is True:
            if self.are_weights_on_device is False:
                self.ul_weights = self.ul_weights.type_as(outputs.logits)
                self.are_weights_on_device = True
            ul_loss = self.unlikelihood_loss(
                decoder_input_ids=batch["decoder_input_ids"],
                weight_mask=self.ul_weights,
                logits=outputs.logits,
                selective_penalty=self.ul_config.ul_selective_penalty,
            )
            ul_loss = self.ul_config.ul_alpha * ul_loss

        if self.ul_config.use_ul is True:
            loss = outputs.loss + ul_loss
            # print(loss)
            return {"loss": loss, "ul_loss": ul_loss.detach()}
        else:
            return {"loss": outputs.loss}

    def rl_step(self, batch):

        log_probs, sample_ids = self.sample_log_probs(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_mask=batch["target_attention_mask"],
            is_greedy=False,
            max_len=self.model_cfg.max_target_len,
        )

        with torch.no_grad():
            _, greedy_ids = self.sample_log_probs(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target_mask=batch["target_attention_mask"],
                is_greedy=True,
                max_len=self.model_cfg.max_target_len,
            )

        # Compute sample reward
        sample_reward_dict = combine_rewards(
            gen=sample_ids,
            tgt=batch["labels"],
            src=batch["input_ids"],
            tokenizer=self.tokenizer,
            decode=True,
            reward_list=self.reward_list,
        )
        sample_rewards = sum(list(sample_reward_dict.values()))
        # s_dict = {f"sample_{k}" : v for k,v in sample_reward_dict.items()}

        # Compute greedy reward
        greedy_reward_dict = combine_rewards(
            gen=greedy_ids,
            tgt=batch["labels"],
            src=batch["input_ids"],
            tokenizer=self.tokenizer,
            decode=True,
            reward_list=self.reward_list,
        )
        greedy_rewards = sum(list(greedy_reward_dict.values()))
        rl_loss = (greedy_rewards - sample_rewards) * log_probs
        rl_loss = torch.mean(rl_loss)

        # if rl_loss < 0.0:
        #    rl_loss = torch.tensor(0.0).type_as(log_probs)

        return {
            "rl_loss": rl_loss,
            "sample_rewards": sample_rewards,
            "greedy_rewards": greedy_rewards,
        }

    def combine_loss(self, mle_loss, rl_loss=None):
        if self.model_cfg.use_rl and rl_loss is not None:
            loss = (
                1 - self.model_cfg.gamma
            ) * mle_loss + self.model_cfg.gamma * rl_loss
            return loss
        else:
            return mle_loss

    def _step(self, batch) -> Dict:
        """Step for training."""
        mle_loss = self.forward(batch)

        if self.model_cfg.use_rl:
            rl_loss = self.rl_step(batch=batch)
            total_loss = self.combine_loss(
                mle_loss=mle_loss["loss"], rl_loss=rl_loss["rl_loss"]
            )
            loss_dict = {
                "loss": total_loss,
                "rl_loss": rl_loss["rl_loss"],
                "sample_rewards": rl_loss["sample_rewards"],
                "greedy_rewards": rl_loss["greedy_rewards"],
            }
            if "ul_loss" in list(mle_loss.keys()):
                loss_dict.update({"ul_loss": mle_loss["ul_loss"]})
            return loss_dict

        else:
            return mle_loss

    def generate_step(self, batch, return_sents=False):
        """Generate step for validation and testing."""
        if self.model_cfg.generate_strat == "greedy":

            gen_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.model_cfg.max_target_len,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )

        elif self.model_cfg.generate_strat == "beam":
            gen_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.model_cfg.max_target_len,
                num_beams=8,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
        elif self.model_cfg.generate_strat == "sample":
            gen_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=self.model_cfg.max_target_len,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
        elif self.model_cfg.generate_strat == "sample_scst":
            _, gen_ids = self.sample_log_probs(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                target_mask=batch["target_attention_mask"],
                is_greedy=False,
                max_len=self.model_cfg.max_target_len,
            )

        sample_reward_dict = combine_rewards(
            gen=gen_ids,
            tgt=batch["labels"],
            src=batch["input_ids"],
            tokenizer=self.tokenizer,
            decode=True,
            reward_list=self.reward_list,
        )
        sample_rewards = sum(list(sample_reward_dict.values()))
        if return_sents:
            return {
                "sample_rewards": sample_rewards,
                "generated": self.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True, cleanup_tokenization_spaces=True
                ),
            }
        else:
            return {"rouge": rouge}

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # op = self.generate_step(batch)
        # Compute sample reward
        _, gen_ids = self.sample_log_probs(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_mask=batch["target_attention_mask"],
            is_greedy=False,
            max_len=self.model_cfg.max_target_len,
        )

        sample_reward_dict = combine_rewards(
            gen=gen_ids,
            tgt=batch["labels"],
            src=batch["input_ids"],
            tokenizer=self.tokenizer,
            decode=True,
            reward_list=self.reward_list,
        )
        sample_rewards = sum(list(sample_reward_dict.values()))
        tqdm_dict = {"val_sample_rewards": sample_rewards}
        self.log_dict(
            tqdm_dict, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True
        )
        return OrderedDict({"progress_bar": tqdm_dict})

    def training_step(self, batch, batch_idx):
        """Training Step"""
        loss_dict = self._step(batch=batch)
        tqdm_dict = {f"train_{k}": v for k, v in loss_dict.items()}
        self.log_dict(tqdm_dict, prog_bar=True, sync_dist=True)
        # return OrderedDict({"loss": loss_dict["loss"], "progress_bar": tqdm_dict})
        return OrderedDict({"loss": loss_dict["loss"]})

    def test_step(self, batch, batch_idx):
        """Test step"""
        op = self.generate_step(batch, return_sents=True)
        return op

    def test_epoch_end(self, outputs):
        """Test epoch end"""
        # sample_rewards = compute_mean(l=outputs, k="sample_rewards")
        sents = [o["generated"] for o in outputs]
        final_op = {"generated": sents}
        self.test_results = [final_op]
        # self.log("test-rouge", rouge, sync_dist=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Return configured schedulers and optimizer."""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.model_cfg.learning_rate)
        tb_size = self.model_cfg.train_batch_size * max(self.node_cfg.max_gpus, 1)
        acc_size = self.model_cfg.accum_steps * float(self.model_cfg.max_epochs)
        self.total_steps = (len(self.train_ds["ds"]) // tb_size) // acc_size
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.total_steps,
        )
        # scheduler = WarmupLR(optimizer, warmup_max_lr=0.0001)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds["ds"],
            batch_size=self.model_cfg.train_batch_size,
            num_workers=self.node_cfg.max_cpus,
            collate_fn=self.train_ds["collate_fn"],
            shuffle=True,
        )

    def val_dataloader(self):
        val_ds = create_data(
            model=self.model, split="valid", start=self.start, stop=self.stop
        )
        return torch.utils.data.DataLoader(
            val_ds["ds"],
            batch_size=self.model_cfg.train_batch_size,
            num_workers=self.node_cfg.max_cpus,
            collate_fn=val_ds["collate_fn"],
            shuffle=False,
        )

    def test_dataloader(self):
        test_ds = create_data(model=self.model, split="test", start=-1, stop=-1)
        return torch.utils.data.DataLoader(
            test_ds["ds"],
            batch_size=self.model_cfg.train_batch_size,
            num_workers=self.node_cfg.max_cpus,
            collate_fn=test_ds["collate_fn"],
            shuffle=False,
        )
