DAT = dat

scripts = $(wildcard ccsd_n=*_l=*_omega=*.py)
targets = $(addprefix $(DAT)/, $(addsuffix .log, $(basename $(scripts))))


all: $(targets)


$(DAT)/%.log: %.py
	python -u $< | tee $@
