# Floor training notes

The experiment used Qwen/Qwen3-8B on Tinker with rank-16 LoRA, learning rate 0.0001, batch size 32, and the `qwen3_disable_thinking` renderer. The tokenizer revision is pinned in `experiment.json`; dependency versions and package hashes are locked in `uv.lock`.

Preparation produced 396 training scenarios and 100 validation scenarios. Each training scenario was rendered with both prompt variants, producing 792 examples per pass. The source mix contained 146 existing human buyer messages and 350 generated scenarios before the train/validation division. Dialogue IDs and generated scenarios were separated, and normalized wording overlapping validation was removed from training. Human text has synthetic seller-policy labels; see the attribution document.

An initial one-pass run was followed by a fresh six-pass run. A checkpoint was saved after each pass. The three-pass checkpoint was selected using the same 20-case development probe:

| System, detailed prompt | Exact interpretation | Raw model action/price | Action/price with shared application policy |
| --- | ---: | ---: | ---: |
| Untuned Qwen3-8B | 5/20 | 9/20 | 12/20 |
| Initial one-pass fine-tune | 11/20 | 9/20 | 18/20 |
| Two total passes | 13/20 | 12/20 | 18/20 |
| Three total passes, selected | 15/20 | 17/20 | 19/20 |
| Four total passes | 14/20 | 16/20 | 18/20 |
| Five total passes | 15/20 | 17/20 | 18/20 |
| Six total passes | 13/20 | 17/20 | 18/20 |

Exact interpretation means all four fields—offer, payment type, seller cost, and intent—match the task labels. The runtime decision metric checks the action and price after applying the same deterministic pricing function to baseline and tuned interpretations. It does not measure complete conversation success or actual sales. The best compact-prompt baseline reached 16/20 runtime decisions.

The selected model's remaining decision error interpreted an installment offer as full payment. The probe also contains an ambiguous generated question/offer label. The same cases were reused to compare checkpoints, so these figures describe development results rather than independent generalization. New conversational context and listing-answer tools were added afterward and were not part of this probe.

The six-pass training run's token cost estimate was approximately $0.80, with roughly $0.01 for its checkpoint comparisons. These figures exclude storage and are local token estimates, not reconciled invoices. More passes did not consistently improve the measured outcomes, so the final six-pass checkpoint was not selected for the demo.

Arithmetic and the final buyer-facing wording come from application code. Fine-tuning is credited for interpreting messages, not for the correctness of deterministic price calculations. The seller can inspect/correct the terms and review the reply before sending it.
