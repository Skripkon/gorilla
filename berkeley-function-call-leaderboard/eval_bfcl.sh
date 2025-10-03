#!/bin/bash

# All supported models are listed in https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/SUPPORTED_MODELS.md

GIGACHAT_MODELS="GigaChat-2-MAX--IFT-FC,GigaChat-2-MAX--PROD-FC"
OPENAI_MODELS="gpt-4o-mini-2024-07-18-FC"
ANTHROPIC_MODELS="claude-3-7-sonnet-20250219-FC" 

TEST_CATEGORIES="\
simple_python,simple_java,simple_javascript,multiple,\
live_simple,live_multiple,\
live_relevance,\
irrelevance,live_irrelevance,\
memory_kv,memory_vector,memory_rec_sum,\
multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context"

NUM_THREADS=10

bfcl generate --model "$GIGACHAT_MODELS" --test-category "$TEST_CATEGORIES" --num-threads $NUM_THREADS
bfcl generate --model "$OPENAI_MODELS" --test-category "$TEST_CATEGORIES" --num-threads $NUM_THREADS
bfcl generate --model "$ANTHROPIC_MODELS" --test-category "$TEST_CATEGORIES" --num-threads $NUM_THREADS

bfcl evaluate --test-category "$TEST_CATEGORIES"  # set --partial-eval if needed
