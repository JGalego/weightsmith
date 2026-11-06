# Workers

Code that runs outside Weightsmith's own environment: rollout workers on this machine, and the benchmark job on Hugging Face Jobs.

## Rollout workers

A `rollouts` probe suite runs episodes of a policy network in a simulated environment. The simulators need dependencies that cannot share Weightsmith's environment (Kinetix needs jax 0.4.38 and its own numpy), so each one runs as a separate process in its own Python environment: Weightsmith writes the policy's weights to a temporary safetensors file and sends the worker one JSON request on stdin; the worker writes one JSON result on stdout (`src/weightsmith/evaluation/rollouts.py`).

Weightsmith looks for a worker's Python at `WEIGHTSMITH_<ENV>_PYTHON`, else at `~/.cache/weightsmith/workers/<env>/venv/bin/python` (`$XDG_CACHE_HOME` is honoured).

### Kinetix

`kinetix_worker.py` runs an actor (`embed` 679 -> 128 with ReLU, `actor0` to `actor4` 128 -> 128 with tanh, `actor_head` 128 -> 16) in one Kinetix level with the symbolic observation and multi-discrete actions, built as Kinetix's own PPO builds it for `env=symbolic env_size=l`. It reports success, red contact, timeout and invalid episodes, the mean return and the mean length.

Set it up once (about 2 minutes, 1.1 GB; CPU only):

```bash
uv venv --python 3.12 ~/.cache/weightsmith/workers/kinetix/venv
VIRTUAL_ENV=~/.cache/weightsmith/workers/kinetix/venv \
  uv pip install --no-deps -r src/weightsmith/workers/kinetix_requirements.txt
```

`--no-deps` matters: the requirements file pins the whole resolved set, and without it Kinetix's metadata would also pull in its training and rendering extras, which the worker never imports.

On a 12-thread laptop CPU, 1,024 episodes take 70 to 155 seconds depending on how long they last, plus 15 to 60 seconds of start-up and compilation (jax's compilation cache lives next to the environment). The worker uses one XLA device per CPU and the legacy XLA:CPU runtime, which were about four times faster here; set `XLA_FLAGS` to override.

Checked against "Sparks of weight-space programming" (Experiment 4, `l/h2_car_ramp`): the trained network and the network as shown to the editor never succeed (every episode times out, mean return 0.152), and the article's edited network succeeds in 1,024 of 1,024 episodes, each 73 steps long with return 1.374, as the article reports.

Differences from the article's description of the environment: an episode is invalid only when the state holds a NaN (Kinetix does not end an episode on an infinity), and when green touches blue and red touches green in different substeps of one step, the first substep decides.

## Benchmark jobs

`hf_bench_job.py` evaluates one shard of a `benchmark` suite as a job on Hugging Face Jobs (`src/weightsmith/evaluation/benchmarks.py` launches it with `hf jobs uv run` semantics: Python 3.12, `vllm==0.32.0`, `math-verify==0.9.0`, `pyarrow==26.0.1`). It downloads the model directory and the patch from the private repository, puts the patch's tensors in place of the originals, samples the answers with vLLM, grades them and uploads `scores.json`, `per_problem.json` and `first_samples.jsonl.gz` (the first answer to every problem) under `results/<key>/`.

Nothing to set up beyond a Hugging Face login with write access (`hf auth login`); the jobs are billed to that account. To follow a job: `hf jobs ps`, `hf jobs logs <id>`; Weightsmith cancels its running jobs when it is interrupted.

The prompt, answer extraction and grading are those of "Sparks of weight-space programming" (`exp8_math/code/evaluate.py` and `mathqa.py` of the article's logs, MIT licence), so shard `i` of 10 holds the same 50 MATH-500 problems as the article's `math500_s<i>of10`.
