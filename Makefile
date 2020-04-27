#TASK = empathetic_dialogues
GPT-SIZE = medium
VTIM = 7200
STIM = 3600
MODEL_PATH = models/$(TASK)/$@/$@

BASE_ARGS = --add-special-tokens True --add-start-token False --gpt2-size $(GPT-SIZE)\
	--inference beam --beam-size 10 --beam-context-block-ngram 3 --beam-block-ngram 3\
	--beam-min-length 25

TRAIN_ARGS = $(MODEL_ARGS) -tblog True -bs 1 -vtim $(VTIM) -stim $(STIM) -vmt ppl -vmm min -mf $(MODEL_PATH) --optimizer \
							adam -lr 6.25e-5

EVAL_ARGS = $(MODEL_ARGS) --save-world-logs True --report-filename eval_results -d True

dialogpt-mc:
	python examples/train_model.py $(TRAIN_ARGS) -t $(TASK) -m hugging_face/dialogpt --next_sentence_prediction True

gpt2:
	python examples/train_model.py $(TRAIN_ARGS) -t $(TASK) -m hugging_face/gpt2 --history-add-global-end-token end

eval:
	python exampels/eval_model.py $(EVAL_ARGS)
		