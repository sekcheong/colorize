TEX = pdflatex -shell-escape -interaction=nonstopmode -file-line-error
BIBTEX = bibtex
JGRAPH = jgraph
THESIS = final

.PHONY: all view

all: $(THESIS).pdf

view:
	open $(THESIS).pdf

$(THESIS).pdf: $(THESIS).bbl $(THESIS).tex
	$(TEX) $(THESIS).tex
	$(TEX) $(THESIS).tex
	open $(THESIS).pdf

$(THESIS).bbl:
	$(TEX) $(THESIS).tex
	$(BIBTEX) $(THESIS)

clean:
	rm -rf *.aux *.lof *.log *.out *.lot *.toc *.bbl *.blg *.pdf *.eps *-eps-converted*.pdf
