default: pdf-docs

pdf-docs: pal-measles.pdf

html-docs: pal-measles.html

%.pdf: %.qmd
	quarto render "$*.qmd" --to pdf

%.html: %.qmd
	quarto render "$*.qmd" --to html

%.html: %.md
	Rscript --vanilla -e "rmarkdown::render(\"$*.md\",output_format=\"html_document\")"

%.R: %.Rmd
	Rscript --vanilla -e "library(knitr); purl(\"$*.Rmd\",output=\"$*.R\")"

clean:
	$(RM) *.o *.so *.log *.aux *.out *.nav *.snm *.toc *.bak
	$(RM) Rplots.ps Rplots.pdf

fresh: clean
	$(RM) -r cache figure

