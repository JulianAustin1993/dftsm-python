# A Makefile for compiling LaTeX documents.
#------------------------------------------------------------------------------
OBJECT		= beamer
NAME		= beamer-pres
BIBTEX		= true
LATEX		= lualatex
TEX_SOURCES	= $(wildcard *.tex) $(wildcard ./*/*.tex)
SOURCES		= $(TEX_SOURCES)
.SUFFIXES: .tex .dvi .ps .pdf .ps.gz .bbl .dat

$(OBJECT).pdf: $(OBJECT).bbl $(OBJECT).tex $(SOURCES)
	$(LATEX) -shell-escape $* --enable-write18
	$(LATEX) -shell-escape $* --enable-write18

$(OBJECT).bbl: $(OBJECT).tex references.bib $(SOURCES)
	$(LATEX) -shell-escape $* --enable-write18
	$(BSTINPUTS) bibtex $*

default: $(OBJECT).pdf

draft: default
	cp $(OBJECT).pdf $(NAME).pdf 
	
tidy:
	rm -f *.aux *.toc *.log *.out  *.nav *.snm *.bbl *.blg *.idx *.lot *.lof */*.aux
clean: tidy
	rm -f $(OBJECT).dvi $(OBJECT).synctex.gz $(OBJECT)-compress.pdf $(OBJECT).pdf $(OBJECT)-figure*.pdf $(OBJECT)-figure*.dep $(OBJECT)-figure*.dpth $(OBJECT)-figure*.table $(OBJECT)-figure*.gnuplot $(OBJECT).ps
compress:
	 gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress -sOutputFile=$(OBJECT)-compress.pdf $(OBJECT).pdf
spell:
	@printf "\033[0;31mSPELL CHECK RESULTS:\033[0m\n"
	@for file in $(TEX_SOURCES); do \
		SPL=$$(cat $$file | aspell list -t --conf=./spell_check/ignore_tex --add-extra-dicts=./spell_check/custom_spellings); \
		if [ "$$SPL" != "" ]; then \
			printf "\033[0;32m$$file\033[0m\n"; \
			echo "$$SPL" | sort | uniq | sed 's/^/  /'; \
		fi \
	done
