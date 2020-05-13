VPATH = models/empathetic_dialogues:models/empathetic_dialogues_mod:models/neil

GPT-SIZE = medium
VTIM = 14400 
STIM = 1800
MODEL_PATH = models/$(TASK)/$@/$@

TRAIN_MODEL = python examples/train_model.py $(TRAIN_ARGS) -t $(TASK)

MODEL_ARGS = -m hugging_face/$(MODEL) -mf $(MODEL_PATH) --add-special-tokens True --add-start-token False --gpt2-size $(GPT-SIZE)\
	--inference topk --topk 10

TRAIN_ARGS = $(MODEL_ARGS) -tblog True -bs 1 -vtim $(VTIM) -stim $(STIM) -vmt token_acc -vmm max --optimizer \
							adam -lr 6.25e-5 -vp 5 -veps 1 --load_from_checkpoint True --update-freq 2 \
							-sval True $(ADDITIONAL_ARGS)

EVAL_ARGS = $(MODEL_ARGS) --save-world-logs True --report-filename eval_results -d True -t $(TASK)

ADDITIONAL_ARGS =

# ED

dialogpt%: MODEL = dialogpt
gpt2%: MODEL = gpt2

%-ed : TASK = empathetic_dialogues
%-neil: TASK = neil

dialogpt_mc%: ADDITIONAL_ARGS += --next_sentence_prediction True
dialogpt_mc_ec%: ADDITIONAL_ARGS += --emotion_prediction True --classes-from-file $(EMOTION_CLASSES_FILE)

dialogpt_mc-ed: TASK = empathetic_dialogues_mod
dialogpt_mc_ec-ed: TASK = empathetic_dialogues_mod
dialogpt_mc_ec-ed: EMOTION_CLASSES_FILE = parlai/tasks/empathetic_dialogues_mod/classes.txt


eval/dialogpt%: MODEL = dialogpt
eval/gpt2%: MODEL = gpt2
eval/%: MODEL_PATH = models/$(TASK)/$*/$*
eval/%:
	python examples/eval_model.py $(EVAL_ARGS)

# ED + Neil

gpt2-ed-neil: gpt2-ed
	$(TRAIN_MODEL) -im $</$(<F)

dialogpt-ed-neil: dialogpt-ed
	$(TRAIN_MODEL) -im $</$(<F)

dialogpt_mc-ed-neil: dialogpt_mc-ed
	$(TRAIN_MODEL) -im $</$(<F)

dialogpt_mc_ec-ed-neil: EMOTION_CLASSES_FILE = parlai/tasks/neil/classes.txt
dialogpt_mc_ec-ed-neil: dialogpt_mc_ec-ed
	$(TRAIN_MODEL) -im -im $</$(<F)

dialogpt_mc_ec-neil: EMOTION_CLASSES_FILE = parlai/tasks/neil/classes.txt

.PHONY: neil
neil: gpt2-neil dialogpt-neil dialogpt_mc-neil dialogpt_mc_ec-neil

.PHONY: ed
ed: gpt2-ed dialogpt-ed dialogpt_mc-ed dialogpt_mc_ec-ed

%: 
	$(TRAIN_MODEL)