.PHONY: all pull pull-blog install prefetch nopyc clean ingest pretrain-tone generate-data train merge export register unregister smoke-test probe setup-base-agent ollama-sweep pipeline pipeline-resume pipeline-stop pipeline-bg _pipeline-launch

LOG_DIR := logs

SHELL := /usr/bin/env bash
PYTHON_BIN      ?= python
PROJECT_VENV    ?= venv
CORTEX_DIR      ?= $(HOME)/Developer/cortex
PRIVATE_DEV_DIR ?= $(HOME)/Developer/private-dev
QUANT           ?= q4_k_m
MODEL_NAME      ?= cortex
INCLUDE_TESTS   ?= false
SAMPLE_EVERY    ?= 1
LIMIT           ?= 0
PIPELINE_SESSION ?= cortex-retrain
PIPELINE_MARK_DIR := $(LOG_DIR)/.pipeline
APPEND        ?=
RESUME        ?=

# Stable Ollama identifiers for the fine-tune base + agent overlay. Keep in
# sync with config.py's BASE_OLLAMA_MODEL / BASE_AGENT_OVERLAY_NAME and with
# tools/init/dev-init-ai. To bump base model versions, change config.py and
# re-pull into Ollama under the same alias (e.g.,
# `ollama pull qwen-next:tag && ollama cp qwen-next:tag qwen`).
BASE_OLLAMA_MODEL       ?= qwen
BASE_AGENT_OVERLAY_NAME ?= qwen-agent

# Set NO_INSTALL=1 to drop install/pull-blog deps from per-stage targets. The
# pipeline sets this after prefetch has populated everything, so each stage
# can run offline. Standalone invocations (`make train`, etc.) still install.
INSTALL_DEP   := $(if $(NO_INSTALL),,install)
PULL_BLOG_DEP := $(if $(NO_INSTALL),,pull-blog)

OFFLINE_ENV := HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_INDEX=1 GIT_TERMINAL_PROMPT=0

all: install

pull:
	@git -C $(CORTEX_DIR) pull --ff-only
	@git -C $(PRIVATE_DEV_DIR) pull --ff-only

pull-blog:
	@git -C $(HOME)/Developer/alexdlaird.github.io pull --ff-only

venv:
	$(PYTHON_BIN) -m pip install virtualenv --user
	$(PYTHON_BIN) -m virtualenv $(PROJECT_VENV)

install: pull venv
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		CUDA_VER=$$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | tr -d '.'); \
		pip install torch --index-url "https://download.pytorch.org/whl/cu$${CUDA_VER}"; \
		pip install -r $(CORTEX_DIR)/finetune/requirements.txt; \
		if [ -d $(HOME)/.unsloth/llama.cpp/gguf-py ]; then \
			pip install --force-reinstall --no-deps $(HOME)/.unsloth/llama.cpp/gguf-py; \
		fi; \
	)

# Front-loads every internet-dependent artifact: git refs, pip wheels, HF model
# weights. Run once before pipeline-bg so the multi-day body runs offline. The
# from_pretrained call mirrors train.py exactly so Unsloth pulls the full set
# of repos it would pull at training time (the source ID, Unsloth's bf16
# mirror, and the bnb-4bit quantized weights).
#
# ollama-sweep is a prereq because the load_in_4bit sanity-allocates the model
# on GPU; if the qwen teacher is still warm, bnb-4bit can't fit 27B in the
# remaining VRAM and dispatches layers to CPU/disk, which 4-bit refuses.
prefetch: install ollama-sweep
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python -c "from unsloth import FastLanguageModel; from config import HF_MODEL_ID, MAX_SEQ_LENGTH; FastLanguageModel.from_pretrained(model_name=HF_MODEL_ID, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=True)"; \
	)

nopyc:
	find . -name '*.pyc' | xargs rm -f || true
	find . -name __pycache__ | xargs rm -rf || true

clean: nopyc
	rm -rf $(PROJECT_VENV)

ingest:
	@$(MAKE) --no-print-directory -C ../rag ingest

pretrain-tone: $(INSTALL_DEP) $(PULL_BLOG_DEP) ollama-sweep
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(CORTEX_DIR)/finetune/pretrain_tone.py; \
	)

generate-data: $(INSTALL_DEP) ollama-sweep
	@mkdir -p $(LOG_DIR)
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(CORTEX_DIR)/finetune/generate_data.py \
			--sample-every $(SAMPLE_EVERY) \
			$(if $(filter-out 0,$(LIMIT)),--limit $(LIMIT),) \
			$(if $(filter true,$(INCLUDE_TESTS)),--include-tests,) \
			$(if $(filter 1,$(APPEND)),--append,); \
	)

train: $(INSTALL_DEP) ollama-sweep
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(CORTEX_DIR)/finetune/train.py \
			$(if $(filter 1,$(RESUME)),--resume,); \
	)

merge: $(INSTALL_DEP) ollama-sweep
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(CORTEX_DIR)/finetune/train.py --merge-only; \
	)

export: $(INSTALL_DEP)
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(CORTEX_DIR)/finetune/export.py --quant $(QUANT) --model-name $(MODEL_NAME); \
	)

setup-base-agent: pull
	@# One-off bootstrap: create $(BASE_AGENT_OVERLAY_NAME) overlay + cortex-agent alias.
	@# Idempotent. Safe to re-run on an already-configured machine.
	$(CORTEX_DIR)/bin/setup-base-agent
	@$(MAKE) --no-print-directory probe

# Sweep every warm model off the GPU and kill any orphaned finetune python
# workers. Used as a prereq to GPU-heavy stages so external clients (Open
# WebUI, opencode, pi) can't strand a model in VRAM and blow the budget.
#
# generate_data.py / pretrain_tone.py / train.py / export.py all daemonize a
# `--bg` worker via subprocess.Popen(start_new_session=True). When tmux is
# killed, the worker survives as an orphan under PID 1 and keeps hitting
# Ollama, which re-warms the teacher model and makes `ollama stop` ineffective
# until the orphan dies. SIGTERM first; SIGKILL the holdouts after a beat.
#
# SKIP_SWEEP=1 short-circuits this target. pipeline-bg sets it for stages
# whose sweep would be redundant (the previous stage already left the GPU
# clean), avoiding a spurious eviction of nomic-embed-text or the qwen
# teacher between in-pipeline stages.
ollama-sweep:
ifeq ($(SKIP_SWEEP),1)
	@echo "==> Skipping ollama-sweep (SKIP_SWEEP=1)."
else
	-@pkill -TERM -f 'cortex/finetune/(generate_data|train|export|pretrain_tone)\.py' 2>/dev/null
	-@sleep 2
	-@pkill -KILL -f 'cortex/finetune/(generate_data|train|export|pretrain_tone)\.py' 2>/dev/null
	-@ollama ps 2>/dev/null | awk 'NR>1 {print $$1}' | xargs -r -n1 ollama stop 2>/dev/null
	-@sleep 1
endif

unregister: ollama-sweep
	-@ollama rm $(MODEL_NAME) 2>/dev/null
	-@ollama rm $(MODEL_NAME)-agent 2>/dev/null
	@# Re-alias the agent name back to the base overlay so pi/opencode keep
	@# working with the same client config after the fine-tune is removed. The
	@# chat-facing $(MODEL_NAME) stays absent; pipeline.py's _resolve_model()
	@# falls back to plain gemma on the chat path.
	ollama cp $(BASE_AGENT_OVERLAY_NAME) $(MODEL_NAME)-agent
	docker restart pipelines

register: ollama-sweep
	@# Clear Ollama's cached manifest/blob digest for this model; otherwise
	@# re-registering an overwritten GGUF trips "digest mismatch".
	-@ollama rm $(MODEL_NAME) 2>/dev/null
	-@ollama rm $(MODEL_NAME)-agent 2>/dev/null
	ollama create $(MODEL_NAME) -f $$($(PROJECT_VENV)/bin/python -c \
		"from config import FINETUNE_OUTPUT_DIR; print(FINETUNE_OUTPUT_DIR / 'gguf' / 'Modelfile')")
	ollama create $(MODEL_NAME)-agent -f $$($(PROJECT_VENV)/bin/python -c \
		"from config import FINETUNE_OUTPUT_DIR; print(FINETUNE_OUTPUT_DIR / 'gguf' / 'Modelfile.agent')")
	@$(MAKE) --no-print-directory smoke-test
	@$(MAKE) --no-print-directory probe
	docker restart pipelines

# Fails if the model doesn't terminate cleanly on a trivial prompt — the
# canonical signature of a broken chat template or missing stop token.
smoke-test:
	@echo "--> smoke-testing $(MODEL_NAME) ..."
	@result=$$(curl -sS --max-time 60 http://localhost:11434/api/chat -d '{ \
		"model":"$(MODEL_NAME)", \
		"messages":[{"role":"user","content":"Reply with the single word: hello"}], \
		"stream":false}'); \
	done_reason=$$(printf '%s' "$$result" | $(PROJECT_VENV)/bin/python -c \
		'import json,sys; print(json.load(sys.stdin).get("done_reason",""))'); \
	if [ "$$done_reason" != "stop" ]; then \
		echo "!!> smoke test FAILED: done_reason='$$done_reason' (expected 'stop')"; \
		printf '%s' "$$result" | head -c 500; echo; \
		exit 1; \
	fi; \
	echo "--> smoke test passed (done_reason=stop)"

# Tool-calling reliability probe against $(MODEL_NAME)-agent. Verifies the
# agent overlay ($(BASE_AGENT_OVERLAY_NAME) or real cortex-agent) emits
# structured tool_calls — the failure mode we saw with temp=1.0 baked into the
# base Modelfile.
probe: pull
	@echo "--> probing $(MODEL_NAME)-agent for tool-calling reliability ..."
	$(CORTEX_DIR)/tests/tool_call_probe.sh $(MODEL_NAME)-agent 10 http://localhost:11434

# Run a pipeline stage with marker tracking. Skips if .done exists (resume);
# leaves a .running marker so pipeline-stop can identify the active stage.
# $1 = stage name (used for marker filename)
# $2 = command to invoke the stage
define run_stage
	@if [ -f $(PIPELINE_MARK_DIR)/$(1).done ]; then \
		echo "==> Skipping $(1) (already complete)."; \
	else \
		mkdir -p $(PIPELINE_MARK_DIR) && touch $(PIPELINE_MARK_DIR)/$(1).running; \
		$(2) && rm -f $(PIPELINE_MARK_DIR)/$(1).running && touch $(PIPELINE_MARK_DIR)/$(1).done; \
	fi

endef

# Full unattended retrain: ingest → generate-data → train → export → register.
# Spawns a detached tmux session running pipeline-bg and tails the log. Ctrl-C
# stops the tail; the tmux session keeps running. Reattach via:
#   tmux attach -t $(PIPELINE_SESSION)
# Override SAMPLE_EVERY for a faster validation pass: make pipeline SAMPLE_EVERY=10
#
# pipeline       — fresh start: clears stage markers and sharegpt.jsonl
# pipeline-resume — picks up where a crashed/stopped run left off; completed
#                   stages are skipped, generate-data appends + dedupes, train
#                   resumes from its HF Trainer checkpoint
# pipeline-stop  — detect current stage; kill if resumable, prompt if not,
#                   refuse if export/register are running (almost done)
pipeline:
	@if tmux has-session -t $(PIPELINE_SESSION) 2>/dev/null; then \
		echo "!!> Pipeline session '$(PIPELINE_SESSION)' is already running."; \
		echo "    Refusing to clear state for a fresh start while a run is live."; \
		echo "    Run 'make pipeline-stop' first, or 'make pipeline-resume' to attach."; \
		exit 1; \
	fi
	@mkdir -p $(LOG_DIR) $(PIPELINE_MARK_DIR)
	@rm -f $(PIPELINE_MARK_DIR)/*.done $(PIPELINE_MARK_DIR)/*.running
	@$(PROJECT_VENV)/bin/python -c "from config import FINETUNE_DATA_DIR; (FINETUNE_DATA_DIR / 'sharegpt.jsonl').unlink(missing_ok=True)" 2>/dev/null || true
	@$(MAKE) --no-print-directory _pipeline-launch

pipeline-resume:
	@mkdir -p $(LOG_DIR) $(PIPELINE_MARK_DIR)
	@$(MAKE) --no-print-directory _pipeline-launch

_pipeline-launch:
	@if tmux has-session -t $(PIPELINE_SESSION) 2>/dev/null; then \
		echo "--> tmux session '$(PIPELINE_SESSION)' already exists; tailing existing log."; \
	else \
		: > $(LOG_DIR)/pipeline.log; \
		tmux new-session -d -s $(PIPELINE_SESSION) "$(MAKE) --no-print-directory pipeline-bg > $(LOG_DIR)/pipeline.log 2>&1"; \
		echo "--> Started pipeline in detached tmux session '$(PIPELINE_SESSION)'."; \
	fi
	@echo "--> Tailing $(LOG_DIR)/pipeline.log (Ctrl-C to stop tailing; pipeline keeps running)."
	@echo "--> Reattach later with: tmux attach -t $(PIPELINE_SESSION)"
	@PANE_PID=$$(tmux list-panes -t $(PIPELINE_SESSION) -F "#{pane_pid}" 2>/dev/null | head -1); \
	  exec tail -f $${PANE_PID:+--pid=$$PANE_PID} $(LOG_DIR)/pipeline.log

pipeline-stop:
	@if ! tmux has-session -t $(PIPELINE_SESSION) 2>/dev/null; then \
		echo "--> No pipeline session running."; \
		exit 0; \
	fi
	@RUNNING=""; for f in $(PIPELINE_MARK_DIR)/*.running; do \
		[ -f "$$f" ] || continue; \
		RUNNING=$$(basename "$$f" .running); break; \
	done; \
	case "$$RUNNING" in \
		merge|export|register) \
			echo "--> Pipeline is in '$$RUNNING' stage — almost complete."; \
			echo "    Refusing to stop; just wait it out (minutes, not hours)."; \
			exit 1; ;; \
		generate-data|train) \
			echo "--> Pipeline is in '$$RUNNING' stage (resumable)."; ;; \
		ingest) \
			echo "!!> Pipeline is in 'ingest' stage (NOT resumable — qdrant rebuilds from scratch)."; \
			printf "    Stop anyway? [y/N] "; read ans </dev/tty || ans=n; \
			[ "$$ans" = "y" ] || { echo "    Aborted."; exit 1; }; ;; \
		*) \
			echo "--> No active stage marker; pipeline is between stages or in prefetch."; \
			printf "    Stop anyway? [y/N] "; read ans </dev/tty || ans=n; \
			[ "$$ans" = "y" ] || { echo "    Aborted."; exit 1; }; ;; \
	esac; \
	tmux kill-session -t $(PIPELINE_SESSION); \
	sleep 2; \
	$(MAKE) --no-print-directory ollama-sweep; \
	echo "--> Pipeline stopped (ollama swept). Resume with 'make pipeline-resume'."

# The actual chain. Halts on first failure. Train and register declare
# ollama-sweep as a prereq, so the teacher model ($(BASE_OLLAMA_MODEL)) and
# any other warm Ollama model are evicted before the HuggingFace 4-bit load.
# Invoked inside the tmux session by `make pipeline` / `make pipeline-resume`.
#
# Prefetch front-loads everything internet-dependent so the body can survive a
# net outage. Each stage runs with NO_INSTALL=1 (skips install/pull deps) and
# OFFLINE_ENV (HF/pip refuse network) to fail fast if something tries to phone
# home. register's probe step intentionally still does git pull — if net is out
# at the very end, re-run `make probe` once it's back.
#
# Each stage is wrapped in run_stage so a .done marker lets pipeline-resume
# skip it next time. generate-data always passes APPEND=1; on a fresh run the
# pipeline target deleted sharegpt.jsonl, so the appender starts clean and
# prepends seeds. On resume, existing pairs are loaded into seen_questions
# for dedupe. train always passes RESUME=1; train.py is a no-op on that flag
# when no checkpoint exists.
#
# merge is its own stage because Unsloth's save_pretrained_merged silently
# no-ops on the post-train model object under recent TRL versions — the
# fine-tuned weights are still there, but they need to be reloaded fresh as a
# peft model in a new process before the merge will actually write anything.
# Splitting it out also makes the pipeline restartable from after train without
# rerunning the long step.
#
# merge runs offline like every other stage; train.py's merge() pre-resolves
# the base model snapshot path from the HF cache before calling
# save_pretrained_merged. Unsloth's saving path can't natively see the
# ~/.cache/huggingface/hub/models--*/ layout — only its loading path can —
# which is why we hand it the resolved snapshot dir explicitly.
#
# pretrain-tone is intentionally not in this chain. It pre-trains a tone
# adapter from blog content; for an agent-style coding assistant, that biases
# toward verbose prose and is off-distribution from real agent use. To opt in,
# run `make pretrain-tone` manually before `make pipeline`.
pipeline-bg: prefetch
	$(call run_stage,ingest,$(OFFLINE_ENV) $(MAKE) --no-print-directory ingest NO_INSTALL=1)
	$(call run_stage,generate-data,$(OFFLINE_ENV) $(MAKE) --no-print-directory generate-data NO_INSTALL=1 APPEND=1 SKIP_SWEEP=1)
	$(call run_stage,train,$(OFFLINE_ENV) $(MAKE) --no-print-directory train NO_INSTALL=1 RESUME=1)
	$(call run_stage,merge,$(OFFLINE_ENV) $(MAKE) --no-print-directory merge NO_INSTALL=1)
	$(call run_stage,export,$(OFFLINE_ENV) $(MAKE) --no-print-directory export NO_INSTALL=1)
	$(call run_stage,register,$(OFFLINE_ENV) $(MAKE) --no-print-directory register NO_INSTALL=1)
