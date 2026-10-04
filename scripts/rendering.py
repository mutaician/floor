"""Use the pinned official Qwen renderer identically for training and sampling."""


def messages(prompt, draft):
    result = [{"role": "system", "content": prompt["system"]}]
    for example in prompt["examples"]:
        result.extend([
            {"role": "user", "content": example["source"]},
            {"role": "assistant", "content": example["target"]},
        ])
    result.append({"role": "user", "content": draft})
    return result


def build_renderer(config):
    from transformers import AutoTokenizer
    from tinker_cookbook import renderers
    from huggingface_hub import snapshot_download
    import os

    options = dict(repo_id=config["model"]["id"], revision=config["model"]["tokenizer_revision"],
        cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
        allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json"])
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        local_path = snapshot_download(**options, local_files_only=True)
    except LocalEntryNotFoundError:
        local_path = snapshot_download(**options)
    tokenizer = AutoTokenizer.from_pretrained(
        local_path, local_files_only=True,
    )
    return renderers.get_renderer(config["model"]["renderer"], tokenizer), tokenizer


def supervised(renderer, prompt, draft, target):
    from tinker_cookbook.renderers import TrainOnWhat

    model_input, weights = renderer.build_supervised_example(
        messages(prompt, draft) + [{"role": "assistant", "content": target}],
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    prefix = renderer.build_generation_prompt(messages(prompt, draft)).to_ints()
    assert model_input.to_ints()[:len(prefix)] == prefix
    # The empty thinking block is already supplied at sampling time. It is
    # framing, so only answer tokens and the final stop token receive loss.
    weights[:len(prefix)] = 0
    return model_input, weights


def count_training_tokens(renderer, prompt, draft, target):
    model_input, _ = supervised(renderer, prompt, draft, target)
    return model_input.length


def verify_mask(renderer, tokenizer, prompt, draft, target):
    model_input, weights = supervised(renderer, prompt, draft, target)
    tokens = model_input.to_ints()
    prefix = renderer.build_generation_prompt(messages(prompt, draft)).to_ints()
    assert tokens[:len(prefix)] == prefix, "Training and sampling renderer prefixes differ"
    assert len(tokens) == len(weights)
    # Verify demonstrations and every sampling-prefix token receive zero loss.
    header = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    starts = [i for i in range(len(tokens) - len(header)) if tokens[i:i + len(header)] == header]
    assert starts
    final_user_start = starts[-1]
    assert not weights[:final_user_start].any(), "Earlier demonstrations carry loss"
    assert not weights[:len(prefix)].any(), "Prompt or draft carries loss"
    trained = [token for token, weight in zip(tokens, weights.tolist()) if weight > 0]
    trained_text = tokenizer.decode(trained)
    assert target in trained_text, "Target is absent from trained tokens"
    generation_text = tokenizer.decode(prefix)
    assert generation_text.endswith("<think>\n\n</think>\n\n")
    assert trained and model_input.length <= 512
    return {
        "renderer": type(renderer).__name__, "train_on_what": "last_assistant_message",
        "complete_sequence_tokens": len(tokens), "sampling_prompt_tokens": len(prefix),
        "trained_tokens": len(trained), "trained_text": trained_text,
        "generation_suffix": generation_text[-65:],
        "demonstration_weights_zero": True, "training_sampling_prefix_identical": True,
    }
