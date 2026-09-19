export NANOCHAT_BASE_DIR="$PWD/Task_5_1_data"

python -m nanochat.dataset -n 8

python -m scripts.tok_train --max-chars=500000000 --vocab-size=8192
mv "$NANOCHAT_BASE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer_8192"

python -m scripts.tok_train --max-chars=500000000 --vocab-size=32768
mv "$NANOCHAT_BASE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/tokenizer_32768"