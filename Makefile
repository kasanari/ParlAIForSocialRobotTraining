#TASK = empathetic_dialogues
GPT-SIZE = medium
VTIM = 7200
STIM = 3600
MODEL_PATH = models/$(TASK)/$@/$@

MODEL_ARGS = -mf $(MODEL_PATH) -im $(MODEL_PATH) --add-special-tokens True --add-start-token False --gpt2-size $(GPT-SIZE)\
	--inference topk --topk 10 

TRAIN_ARGS = $(MODEL_ARGS) -tblog True -bs 1 -vtim $(VTIM) -stim $(STIM) -vmt ppl -vmm min --optimizer \
							adam -lr 6.25e-5 -vp 3 -veps 1 --load_from_checkpoint True

EVAL_ARGS = $(MODEL_ARGS) --save-world-logs True --report-filename eval_results -d True

dialogpt-mc:
	python examples/train_model.py $(TRAIN_ARGS) -t $(TASK) -m hugging_face/dialogpt --next_sentence_prediction True

gpt2:
	python examples/train_model.py $(TRAIN_ARGS) -t $(TASK) -m hugging_face/gpt2 --history-add-global-end-token end

eval:
	python exampels/eval_model.py $(EVAL_ARGS)
		