DAT = dat

scripts = $(wildcard ccsd_n=*_l=132_omega=*.py)
targets = $(addprefix $(DAT)/, $(addsuffix .log, $(basename $(scripts))))


scripts_90 = $(wildcard ccsd_n=*_l=90_omega=*.py)
targets_90 = $(addprefix $(DAT)/, $(addsuffix .log, $(basename $(scripts_90))))


all: $(targets_90)


$(DAT)/%.log: %.py
	python -u $< | tee $@
