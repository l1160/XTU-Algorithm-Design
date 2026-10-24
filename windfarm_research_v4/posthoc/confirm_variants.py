"""Explicit proposed LM population-retention adaptation; not original LMOCSO."""
import numpy as np
import algorithms as A
import variants as V
ORIGINAL_LM_SELECT=A.lm_select
ORIGINAL_CONFIGURE=V.configure

def fill_select(pop,refs,theta):
    # Keep every original APD survivor, then fill from existing candidates.
    # No new evaluation, random immigrant, or phenotype deduplication is added.
    first=A.fronts(pop)[0]
    primary_ids=first[A.apd_ids(pop.take(first),refs,theta)]
    target=min(100,len(pop))
    assert len(primary_ids)<=target
    if len(primary_ids)==target:return pop.take(primary_ids)
    rest=np.setdiff1d(np.arange(len(pop)),primary_ids,assume_unique=True)
    extra=A.gde_select(pop.take(rest),target-len(primary_ids))
    return A.join(pop.take(primary_ids),extra)

def configure(options):
    ORIGINAL_CONFIGURE(options)
    A.lm_select=fill_select if options.get("fill_population") else ORIGINAL_LM_SELECT

def install():
    V.VARIANTS["LM_fill_population"]=dict(method="LMOCSO",fill_population=True)
    V.configure=configure
