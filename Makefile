figure_deps = $(wildcard, results/benchmarks/zanghellini/dat/*) \
			  $(wildcard, results/benchmarks/li/dat/*) \
			  $(wildcard, results/benchmarks/miyagi/dat/*)

all:
	make build
	make build

main.makefile: $(figure_deps)
	lualatex -shell-escape main.tex

build: main.makefile
	make -j4 -f main.makefile
	latexmk -lualatex -bibtex -shell-escape

watch: main.makefile
	make -j4 -f main.makefile
	latexmk -lualatex -bibtex -shell-escape -pvc -g -view=none

p:
	make presentation

presentation:
	latexmk -lualatex -bibtex -shell-escape -g presentation


.PHONY: clean clean-figures


clean:
	latexmk -C
	$(RM) main.bbl main.makefile main.figlist main.run.xml betterbib_cache.sqlite


clean-figures:
	$(RM) figures/main-figure*
