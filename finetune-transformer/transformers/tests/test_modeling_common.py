# coding=utf-8
# Copyright 2019 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import logging
import os.path
import random
import tempfile
import unittest

from transformers import is_torch_available

from .utils import require_torch, slow, torch_device


if is_torch_available():
    import torch
    import numpy as np

    from transformers import (
        AdaptiveEmbedding,
        PretrainedConfig,
        PreTrainedModel,
        BertModel,
        BertConfig,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
    )


def _config_zero_init(config):
    configs_no_init = copy.deepcopy(config)
    for key in configs_no_init.__dict__.keys():
        if "_range" in key or "_std" in key or "initializer_factor" in key:
            setattr(configs_no_init, key, 0.0)
    return configs_no_init


@require_torch
class ModelTesterMixin:

    model_tester = None
    all_model_classes = ()
    test_torchscript = True
    test_pruning = True
    test_resize_embeddings = True
    test_head_masking = True
    is_encoder_decoder = False

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            out_2 = outputs[0].numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    after_outputs = model(**inputs_dict)

                # Make sure we don't have nans
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        param.data.mean().item(),
                        [0.0, 1.0],
                        msg="Parameter {} of model {} seems not properly initialized".format(name, model_class),
                    )

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                first = model(**inputs_dict)[0]
                second = model(**inputs_dict)[0]
            out_1 = first.cpu().numpy()
            out_2 = second.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        decoder_seq_length = (
            self.model_tester.decoder_seq_length
            if hasattr(self.model_tester, "decoder_seq_length")
            else self.model_tester.seq_length
        )
        encoder_seq_length = (
            self.model_tester.encoder_seq_length
            if hasattr(self.model_tester, "encoder_seq_length")
            else self.model_tester.seq_length
        )
        decoder_key_length = (
            self.model_tester.key_length if hasattr(self.model_tester, "key_length") else decoder_seq_length
        )
        encoder_key_length = (
            self.model_tester.key_length if hasattr(self.model_tester, "key_length") else encoder_seq_length
        )

        for model_class in self.all_model_classes:
            config.output_attentions = True
            config.output_hidden_states = False
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            attentions = outputs[-1]
            self.assertEqual(model.config.output_attentions, True)
            self.assertEqual(model.config.output_hidden_states, False)
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                self.assertEqual(out_len % 2, 0)
                decoder_attentions = outputs[(out_len // 2) - 1]
                self.assertEqual(model.config.output_attentions, True)
                self.assertEqual(model.config.output_hidden_states, False)
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

            # Check attention is always last and order is fine
            config.output_attentions = True
            config.output_hidden_states = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            self.assertEqual(out_len + (2 if self.is_encoder_decoder else 1), len(outputs))
            self.assertEqual(model.config.output_attentions, True)
            self.assertEqual(model.config.output_hidden_states, True)

            self_attentions = outputs[-1]
            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    def test_torchscript(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        self._create_and_check_torchscript(config, inputs_dict)

    def test_torchscript_output_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        config.output_attentions = True
        self._create_and_check_torchscript(config, inputs_dict)

    def test_torchscript_output_hidden_state(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        config.output_hidden_states = True
        self._create_and_check_torchscript(config, inputs_dict)

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = inputs_dict["input_ids"]  # Let's keep only input_ids

            try:
                traced_gpt2 = torch.jit.trace(model, inputs)
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_gpt2, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_params = model.parameters()
            loaded_model_params = loaded_model.parameters()

            models_equal = True
            for p1, p2 in zip(model_params, loaded_model_params):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_headmasking(self):
        if not self.test_head_masking:
            return

        global_rng.seed(42)
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        global_rng.seed()

        config.output_attentions = True
        config.output_hidden_states = True
        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            # Prepare head_mask
            # Set require_grad after having prepared the tensor to avoid error (leaf variable has been moved into the graph interior)
            head_mask = torch.ones(
                self.model_tester.num_hidden_layers, self.model_tester.num_attention_heads, device=torch_device
            )
            head_mask[0, 0] = 0
            head_mask[-1, :-1] = 0
            head_mask.requires_grad_(requires_grad=True)
            inputs = inputs_dict.copy()
            inputs["head_mask"] = head_mask

            outputs = model(**inputs)

            # Test that we can get a gradient back for importance score computation
            output = sum(t.sum() for t in outputs[0])
            output = output.sum()
            output.backward()
            multihead_outputs = head_mask.grad

            attentions = outputs[-1]

            # Remove Nan
            for t in attentions:
                self.assertLess(
                    torch.sum(torch.isnan(t)), t.numel() / 4
                )  # Check we don't have more than 25% nans (arbitrary)
            attentions = [
                t.masked_fill(torch.isnan(t), 0.0) for t in attentions
            ]  # remove them (the test is less complete)

            self.assertIsNotNone(multihead_outputs)
            self.assertEqual(len(multihead_outputs), self.model_tester.num_hidden_layers)
            self.assertAlmostEqual(attentions[0][..., 0, :, :].flatten().sum().item(), 0.0)
            self.assertNotEqual(attentions[0][..., -1, :, :].flatten().sum().item(), 0.0)
            self.assertNotEqual(attentions[1][..., 0, :, :].flatten().sum().item(), 0.0)
            self.assertAlmostEqual(attentions[-1][..., -2, :, :].flatten().sum().item(), 0.0)
            self.assertNotEqual(attentions[-1][..., -1, :, :].flatten().sum().item(), 0.0)

    def test_head_pruning(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            config.output_attentions = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            heads_to_prune = {0: list(range(1, self.model_tester.num_attention_heads)), -1: [0]}
            model.prune_heads(heads_to_prune)
            with torch.no_grad():
                outputs = model(**inputs_dict)

            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_pretrained(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            config.output_attentions = True
            config.output_hidden_states = False
            model = model_class(config=config)
            model.to(torch_device)
            model.eval()
            heads_to_prune = {0: list(range(1, self.model_tester.num_attention_heads)), -1: [0]}
            model.prune_heads(heads_to_prune)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name)
                model.to(torch_device)

            with torch.no_grad():
                outputs = model(**inputs_dict)
            attentions = outputs[-1]
            self.assertEqual(attentions[0].shape[-3], 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_save_load_from_config_init(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            config.output_attentions = True
            config.output_hidden_states = False

            heads_to_prune = {0: list(range(1, self.model_tester.num_attention_heads)), -1: [0]}
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**inputs_dict)
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[-1].shape[-3], self.model_tester.num_attention_heads - 1)

    def test_head_pruning_integration(self):
        if not self.test_pruning:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            if "head_mask" in inputs_dict:
                del inputs_dict["head_mask"]

            config.output_attentions = True
            config.output_hidden_states = False

            heads_to_prune = {0: [0], 1: [1, 2]}
            config.pruned_heads = heads_to_prune

            model = model_class(config=config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**inputs_dict)
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

            with tempfile.TemporaryDirectory() as temp_dir_name:
                model.save_pretrained(temp_dir_name)
                model = model_class.from_pretrained(temp_dir_name)
                model.to(torch_device)

            with torch.no_grad():
                outputs = model(**inputs_dict)
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads)
            self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

            heads_to_prune = {0: [0], 2: [1, 2]}
            model.prune_heads(heads_to_prune)

            with torch.no_grad():
                outputs = model(**inputs_dict)
            attentions = outputs[-1]

            self.assertEqual(attentions[0].shape[-3], self.model_tester.num_attention_heads - 1)
            self.assertEqual(attentions[1].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[2].shape[-3], self.model_tester.num_attention_heads - 2)
            self.assertEqual(attentions[3].shape[-3], self.model_tester.num_attention_heads)

            self.assertDictEqual(model.config.pruned_heads, {0: [0], 1: [1, 2], 2: [1, 2]})

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config.output_hidden_states = True
            config.output_attentions = False
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            hidden_states = outputs[-1]
            self.assertEqual(model.config.output_attentions, False)
            self.assertEqual(model.config.output_hidden_states, True)
            self.assertEqual(len(hidden_states), self.model_tester.num_hidden_layers + 1)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [
                    self.model_tester.encoder_seq_length
                    if hasattr(self.model_tester, "encoder_seq_length")
                    else self.model_tester.seq_length,
                    self.model_tester.hidden_size,
                ],
            )

    def test_resize_tokens_embeddings(self):
        original_config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            model(**inputs_dict)

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Check that the model can still do a forward pass successfully (every parameter should be resized)
            # Input ids should be clamped to the maximum size of the vocabulary
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 1)
            model(**inputs_dict)

            # Check that adding and removing tokens has not modified the first part of the embedding matrix.
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (torch.nn.Embedding, AdaptiveEmbedding))
            model.set_input_embeddings(torch.nn.Embedding(10, 10))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, torch.nn.Linear))

    def test_tie_model_weights(self):
        if not self.test_torchscript:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_same_values(layer_1, layer_2):
            equal = True
            for p1, p2 in zip(layer_1.weight, layer_2.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    equal = False
            return equal

        for model_class in self.all_model_classes:
            config.torchscript = True
            model_not_tied = model_class(config)
            if model_not_tied.get_output_embeddings() is None:
                continue

            params_not_tied = list(model_not_tied.parameters())

            config_tied = copy.deepcopy(config)
            config_tied.torchscript = False
            model_tied = model_class(config_tied)
            params_tied = list(model_tied.parameters())

            # Check that the embedding layer and decoding layer are the same in size and in value
            self.assertGreater(len(params_not_tied), len(params_tied))
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # embeddings.weight.data.div_(2)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # # Check that after modification, they remain the same.
            # decoding.weight.data.div_(4)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(embeddings.weight.shape, decoding.weight.shape)
            # self.assertTrue(check_same_values(embeddings, decoding))

            # Check that after resize they remain tied.
            model_tied.resize_token_embeddings(config.vocab_size + 10)
            params_tied_2 = list(model_tied.parameters())
            self.assertGreater(len(params_not_tied), len(params_tied))
            self.assertEqual(len(params_tied_2), len(params_tied))

            # decoding.weight.data.mul_(20)
            # # Check that the embedding layer and decoding layer are the same in size and in value
            # self.assertTrue(model.transformer.wte.weight.shape, model.lm_head.weight.shape)
            # self.assertTrue(check_same_values(model.transformer.wte, model.lm_head))

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.is_encoder_decoder:
            input_ids = inputs_dict["input_ids"]
            del inputs_dict["input_ids"]
        else:
            encoder_input_ids = inputs_dict["encoder_input_ids"]
            decoder_input_ids = inputs_dict["decoder_input_ids"]
            del inputs_dict["encoder_input_ids"]
            del inputs_dict["decoder_input_ids"]

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs_dict["inputs_embeds"] = wte(input_ids)
            else:
                inputs_dict["encoder_inputs_embeds"] = wte(encoder_input_ids)
                inputs_dict["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs_dict)


global_rng = random.Random()


def ids_tensor(shape, vocab_size, rng=None, name=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return torch.tensor(data=values, dtype=torch.float, device=torch_device).view(shape).contiguous()


@require_torch
class ModelUtilsTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        logging.basicConfig(level=logging.INFO)
        for model_name in list(BERT_PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            config = BertConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, PretrainedConfig)

            model = BertModel.from_pretrained(model_name)
            model, loading_info = BertModel.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, PreTrainedModel)
            for value in loading_info.values():
                self.assertEqual(len(value), 0)

            config = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            self.assertEqual(model.config.output_attentions, True)
            self.assertEqual(model.config.output_hidden_states, True)
            self.assertEqual(model.config, config)
