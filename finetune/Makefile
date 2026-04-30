.PHONY: all pull pull-blog install prefetch nopyc clean ingest pretrain-tone generate-data train export register unregister smoke-test probe setup-base-agent pipeline pipeline-bg

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
# from_pretrained call mirrors train.py/pretrain_tone.py exactly so Unsloth
# pulls the full set (google/gemma-4-31B-it, unsloth/gemma-4-31b-it, and the
# unsloth/gemma-4-31b-it-unsloth-bnb-4bit quantized weights, ~136GB total).
prefetch: install pull-blog
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

pretrain-tone: $(INSTALL_DEP) $(PULL_BLOG_DEP)
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(CORTEX_DIR)/finetune/pretrain_tone.py; \
	)

generate-data: $(INSTALL_DEP)
	@mkdir -p $(LOG_DIR)
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(CORTEX_DIR)/finetune/generate_data.py \
			--sample-every $(SAMPLE_EVERY) \
			$(if $(filter-out 0,$(LIMIT)),--limit $(LIMIT),) \
			$(if $(filter true,$(INCLUDE_TESTS)),--include-tests,); \
	)

train: $(INSTALL_DEP)
	@( \
		source $(PROJECT_VENV)/bin/activate; \
		python $(CORTEX_DIR)/finetune/train.py; \
	)

train-merge: $(INSTALL_DEP)
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
	@# One-off bootstrap: create gemma4-agent overlay + cortex-agent alias.
	@# Idempotent. Safe to re-run on an already-configured machine.
	$(CORTEX_DIR)/bin/setup-base-agent
	@$(MAKE) --no-print-directory probe

unregister:
	-@ollama rm $(MODEL_NAME) 2>/dev/null
	-@ollama rm $(MODEL_NAME)-agent 2>/dev/null
	@# Re-alias the agent name back to the base overlay so pi/opencode keep
	@# working with the same client config after the fine-tune is removed. The
	@# chat-facing $(MODEL_NAME) stays absent; pipeline.py's _resolve_model()
	@# falls back to plain gemma4 on the chat path.
	ollama cp gemma4-agent $(MODEL_NAME)-agent
	docker restart pipelines

register:
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

# Tool-calling reliability probe against $(MODEL_NAME)-agent. Verifies the agent
# overlay (gemma4-agent or real cortex-agent) emits structured tool_calls — the
# failure mode we saw with temp=1.0 baked into the base Modelfile.
probe: pull
	@echo "--> probing $(MODEL_NAME)-agent for tool-calling reliability ..."
	$(CORTEX_DIR)/tests/tool_call_probe.sh $(MODEL_NAME)-agent 10 http://localhost:11434

# Full unattended retrain: ingest → pretrain-tone → generate-data → train → export → register.
# Spawns a detached tmux session running pipeline-bg and tails the log. Ctrl-C
# stops the tail; the tmux session keeps running. Reattach via:
#   tmux attach -t $(PIPELINE_SESSION)
# Override SAMPLE_EVERY for a faster validation pass: make pipeline SAMPLE_EVERY=10
pipeline:
	@mkdir -p $(LOG_DIR)
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

# The actual chain. Halts on first failure. Ollama models are evicted between
# Ollama-using and HuggingFace-using stages to avoid VRAM contention with the
# 31B Gemma 4 load. Invoked inside the tmux session by `make pipeline`.
#
# Prefetch front-loads everything internet-dependent so the body can survive a
# net outage. Each stage runs with NO_INSTALL=1 (skips install/pull deps) and
# OFFLINE_ENV (HF/pip refuse network) to fail fast if something tries to phone
# home. register's probe step intentionally still does git pull — if net is out
# at the very end, re-run `make probe` once it's back.
pipeline-bg: prefetch
	$(OFFLINE_ENV) $(MAKE) --no-print-directory ingest NO_INSTALL=1
	-@ollama stop nomic-embed-text 2>/dev/null
	$(OFFLINE_ENV) $(MAKE) --no-print-directory pretrain-tone NO_INSTALL=1
	$(OFFLINE_ENV) $(MAKE) --no-print-directory generate-data NO_INSTALL=1
	-@ollama stop gemma4 2>/dev/null
	$(OFFLINE_ENV) $(MAKE) --no-print-directory train NO_INSTALL=1
	$(OFFLINE_ENV) $(MAKE) --no-print-directory export NO_INSTALL=1
	$(OFFLINE_ENV) $(MAKE) --no-print-directory register NO_INSTALL=1
