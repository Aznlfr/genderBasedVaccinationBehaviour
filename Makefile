#
current: target
-include target.mk
Ignore = target.mk

pyvenv: ; $(cleanpyvenv)
-include makestuff/pyvenv.mk
-include makestuff/python.def
-include makestuff/perl.def

vim_session:
	bash -ic "vmt"

-include makestuff/python.def

######################################################################

Sources += $(wildcard *.tex *.bib)


######################################################################

Sources += $(wildcard Codes/*.R)
Sources += $(wildcard Codes/*.py)
autopipeR = defined

slowtarget/synthetic_code_python.out: codes/synthetic_estimation.py
	$(PITH)
######################################################################
### Makestuff

Sources += Makefile

Ignore += makestuff
msrepo = https://github.com/dushoff

Makefile: makestuff/01.stamp
makestuff/%.stamp: | makestuff
	- $(RM) makestuff/*.stamp
	cd makestuff && $(MAKE) pull
	touch $@
makestuff:
	git clone --depth 1 $(msrepo)/makestuff

-include makestuff/os.mk
-include makestuff/pipeR.mk
-include makestuff/visual.mk
-include makestuff/texj.mk
-include makestuff/slowtarget.mk
-include makestuff/pandoc.mk
-include makestuff/git.mk
