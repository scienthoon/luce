# Data generation and annotation

`luce synth` supports generated inputs, existing inputs, selective annotation, and additions to an existing dataset. It writes the decision-record format used by training: one state and one question per JSONL row.

## Modes

| `--mode` | Input source | Label behavior |
|---|---|---|
| `new` (default) | Writer generates states | Teacher labels selected questions |
| `label` | `--input FILE` | Annotate missing selected labels; retain existing labeled records |
| `relabel` | `--input FILE` | Replace selected labels; retain other question records |
| `append` | Generate states, or use `--input FILE` | Add records to existing `train.jsonl` / `val.jsonl` while retaining previous rows and split membership |

`label` and `relabel` use the Teacher without generating personas or new states. Existing label and probability fields are not included in relabeling prompts. The source input file is read rather than rewritten; choose the destination with `--out`.

`append` retains existing rows and adds newly produced rows, including results whose state and question already appear in the output. Existing labels are preserved alongside the new rows. Running append again adds the repeated results again. An explicit count describes additions and must still be satisfied; otherwise the run fails without replacing existing files.

```bash
luce synth --mode new --questions route,risk --counts route=600,risk=200 --out data/generated
luce synth --mode label --input game_events.jsonl --questions action --out data/game
luce synth --mode relabel --input old_labels.jsonl --questions route --out data/revised
luce synth --mode append --input fresh_events.jsonl --out data/game
```

## Question selection and record counts

- `--questions route,risk` selects configured question names.
- `--types choice,noul` selects question types. When both selectors are present, a question must satisfy both.
- `--n 500` requests a shared pool of 500 new states, normally one record per selected question per state. For imported inputs, all available input groups are processed unless `--n` is explicitly set; a limit restricts annotation and preserves previously labeled records outside that prefix.
- `--counts route=600,risk=200` overrides individual selected-question record targets. Counts include both the new train and validation records, not LLM calls or tokens. For imported inputs, they count newly labeled or replaced records; retained existing labels are additional output rows. The generated state pool can grow to meet the largest target.
- A count of zero excludes that question from annotation, including any dynamic-candidate requirements.
- In append mode, targets describe additions; old rows do not count as newly produced records.
- `--max-rounds 3` bounds attempts to replace failed or incorrectly labeled generated examples. If required targets cannot be met, the run fails rather than treating an underfilled dataset as complete.

Questions on the same shared input stay in the same split. More votes, difficult answer targets, and replenishment can increase the number of API calls. Generated states proceed to annotation even when their content repeats another state or a seed example.

Generation uses one shared state pool; there is no independent-state mode per question. The Teacher labels all selected questions against those inputs, and smaller per-question counts select fewer saved rows. Conditional execution at serving time remains in your application code.

## Configuration

The same controls can be saved in `luce.yaml`. CLI values override the corresponding config values.

```yaml
task:
  name: support
  description: Route a support request and estimate its urgency.

questions:
  route:
    type: choice
    prompt: Which queue handles this request?
    options:
      billing: Payments, refunds, and duplicate charges
      shipping: Shipping and delivery
  risk:
    type: score
    prompt: How urgent is this request?
    levels: [Low, Normal, High]

synth:
  mode: new
  n: 500
  questions: [route, risk]
  counts:
    route: 500
    risk: 200
  teacher: "http://localhost:11434/v1|qwen2.5:7b"
  max_rounds: 3
  grid:
    language: {ko: 70, en: 30}
    completeness: [complete, missing_detail]
  val_fraction: 0.1
```

`synth.input` supplies the import path, `synth.types` supplies the type selector, and `synth.writer` can choose a separate model for state generation. A grid axis may be an equally weighted list or a value-to-weight mapping. Weights are relative and need not sum to 100. Grid axes are combined; a weighted `language` axis and an equal `completeness` axis express independent proportions, not an arbitrary joint distribution. Grid weights allocate newly generated shared states; imported text is labeled as provided. Smaller question quotas select subsets of the shared pool.

```bash
luce synth --grid-ratios '{"language":{"ko":70,"en":30}}' --dry-run
```

## Natural labels and optional answer ratios

The default writer is not told to make a correct answer occur at a predetermined rate. The Teacher labels the resulting states, preserving the generator's observed answer mix.

Explicit `answer_ratios` can request a training mixture for fixed-option Choice, Score, or Noul questions:

```yaml
synth:
  answer_ratios:
    route: {billing: 60, shipping: 40}
    risk: {"0": 20, "1": 50, "2": 30}
```

For Noul, use the quoted keys `"true"` and `"false"`. Supply every possible answer key for each configured ratio. Integer quotas use largest-remainder rounding, so very small datasets can only approximate ratios. With new generated inputs, answer targets guide training-state creation, but the Teacher must independently confirm each target; its label is never overwritten to fill a quota. With imported inputs, ratios select from newly annotated or replaced training labels. An explicit count fails if the input cannot supply that mixture; without an explicit count, those labels are downsampled to a feasible mixture. Existing labels retained by `label` mode are not rebalanced, so a requested ratio does not describe the complete output when retained rows are also present. Dynamic candidate questions cannot use global answer ratios because their keys have per-input meanings.

Answer ratios apply to training records. Validation states are reserved independently and retain the unbalanced Teacher-label distribution. Keep a separate evaluation set with representative real labels. **Changing class frequencies changes the probability problem; temperature scaling alone does not guarantee correction of changed base rates.**

## Existing inputs and changing candidates

You may supply decision records with missing labels. Preserve a stable question identity with `meta.question_name`:

```json
{"state":{"hp":20,"nearby_enemies":[{"id":"e1","distance":3}],"can_heal":false},"type":"choice","question":"Which action should be taken?","options":{"dodge":"Dodge the enemy","attack":"Attack enemy e1"},"meta":{"question_name":"action"}}
```

The question name must refer to a question in your config. JSON numeric, boolean, nested object, and array values are preserved; importing a state does not convert it into an author's text description.

Repeated raw input lines remain separate observations. Decision rows for different questions share an observation by state and optional `record_id`/`state_id`/`id`; repeated occurrences of the same question form separate observations. Every occurrence is retained for labeling or relabeling.

For a Choice whose candidates vary with each input, configure:

```yaml
task:
  name: combat
  description: Choose an action from the available candidates.
  state:
    fields: [hp, enemy_distance, can_heal]

questions:
  action:
    type: choice
    prompt: Which action should be taken?
    dynamic_options: true
    options_prompt: Supply two or more legal actions for this state, using stable local IDs and concrete descriptions.
```

When generating states, the writer returns an envelope containing the state and a candidate map for each dynamic question. The same envelope is accepted as imported data:

```json
{"state":{"hp":20,"enemy_distance":3,"can_heal":false},"options":{"action":{"dodge":"Dodge the nearby enemy","attack":"Attack the nearby enemy"}}}
{"state":{"hp":90,"enemy_distance":50,"can_heal":true},"options":{"action":{"approach":"Move closer to the enemy","wait":"Stay in cover"}}}
```

The Teacher sees those per-state candidates, and each output record retains its own candidate map. Missing or invalid required candidate sets are rejected. Existing labels, including labels supplied by a simulator, can be provided directly in the decision format.

## Preview and inspect

Every mode supports `--dry-run`: it can read configuration and input files to plan the work, but it does not call models or write output files.

```bash
luce synth --mode label --input events.jsonl --questions route --dry-run
```

Inspect `synth_report.json` alongside the resulting `train.jsonl` and `val.jsonl` for achieved question counts, label distributions, and generation failures. Grid weights allocate requested prompt attributes.
