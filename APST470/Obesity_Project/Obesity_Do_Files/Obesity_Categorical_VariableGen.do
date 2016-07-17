*Negative Behaviors*
corr UNINSURE_Rate FEW_FruitVeg_Rate  No_EXER_Rate  SMOKER_Rate
factor UNINSURE_Rate FEW_FruitVeg_Rate  No_EXER_Rate  SMOKER_Rate
rotate
egen NegBehav = rowmean (FEW_FruitVeg_Rate No_EXER_Rate SMOKER_Rate)

*Proactive Behaviors Related Health, but not Obesity*
corr  MammogrmRate Pap_SmearRate ProctoscopRate
factor MammogrmRate Pap_SmearRate ProctoscopRate
rotate
egen ProBehav = rowmean ( MammogrmRate Pap_SmearRate ProctoscopRate)

*Health Provider Presence*
corr DENT_Rateper100 PHYS_Rateper100
factor  DENT_Rateper100 PHYS_Rateper100
rotate
egen ProvidePresent = rowmean (  DENT_Rateper100 PHYS_Rateper100)

*Health Structure Presence*
corr CenterYN HP_ShortageY0N1
factor CenterYN HP_ShortageY0N1
rotate
egen StructPresent = rowmean ( CenterYN HP_ShortageY0N1)



*Disadvantaged Populations in terms of ability to care for self*
corr DIABETES_Rate Disabl_Medcar_Rate Elder_Medcar_Rate Hi_BPRate MAJ_Depress_Rate SEV_Work_Disabl_Rate UNEMPL_Rate
factor DIABETES_Rate Disabl_Medcar_Rate Elder_Medcar_Rate Hi_BPRate MAJ_Depress_Rate SEV_Work_Disabl_Rate UNEMPL_Rate
rotate
egen HlthDisadvtg = rowmean ( DIABETES_Rate Disabl_Medcar_Rate Elder_Medcar_Rate Hi_BPRate MAJ_Depress_Rate SEV_Work_Disabl_Rate UNEMPL_Rate)

aorder
