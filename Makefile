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
	latexmk -lualatex -bibtex -shell-escape -pvc -view=none
