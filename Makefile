figure_deps = $(wildcard, dat/*)

all:
	make build

main.makefile: $(figure_deps)
	lualatex -shell-escape main.tex

build: main.makefile
	make -j4 -f main.makefile
	latexmk -lualatex -shell-escape
