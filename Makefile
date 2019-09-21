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

presentation.makefile:
	lualatex -shell-escape presentation.tex

buildp: presentation.makefile
	make -j4 -f presentation.makefile
	latexmk -lualatex -bibtex -shell-escape

p: presentation.makefile
	make -j4 -f presentation.makefile
	latexmk -lualatex -bibtex -shell-escape -g -pvc presentation


.PHONY: clean clean-figures


clean:
	latexmk -C
	$(RM) main.bbl main.makefile main.figlist main.run.xml betterbib_cache.sqlite
	$(RM) presentation.bbl presentation.makefile presentation.figlist presentation.run.xml


clean-figures:
	$(RM) figures/main-figure*
