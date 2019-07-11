figure_deps = $(wildcard, results/benchmarks/zanghellini/dat/*)

all:
	make build

main.makefile: $(figure_deps)
	lualatex -shell-escape main.tex

build: main.makefile
	make -j4 -f main.makefile
	latexmk -lualatex -bibtex -shell-escape
