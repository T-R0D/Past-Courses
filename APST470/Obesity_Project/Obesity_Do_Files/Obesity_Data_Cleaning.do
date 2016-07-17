	/*Rename simply to be easily recognized*/

rename  areaname County_Area
rename  percentofpersonswithlessthanhigh EDUlessthanHSprcnt
rename  percentofpersonswithonlyahighsch EDUonlyHSprcnt
rename  percentofpersonscompletingsomeco EDUsomeCollegeprcnt
rename  percentofpersonswithacollegedegr EDUcollDegreeprcnt
rename  age_19_under AGEunder_19
rename  age_19_64 AGE19_64
rename  age_65_84 AGE65_84
rename  age_85_and_over AGEover_85
rename  incomereportedyn IncomeReportYN

	/*Recoding variables to eliminate missing data codes*/

recode obesity (-1111.1=.), gen (Obesity_Rate)
recode no_exercise (-1111.1=.), gen ( No_EXER_Rate)
recode few_fruit_veg (-1111.1=.), gen ( FEW_FruitVeg_Rate)
recode high_blood_pres (-1111.1=.), gen ( Hi_BPRate)
recode smoker (-1111.1=.), gen ( SMOKER_Rate)
recode diabetes (-1111.1=.), gen ( DIABETES_Rate)
recode elderly_medicare (-2222=.), gen ( elder_medicareRec)
recode disabled_medicare (-2222=.), gen ( disabl_medicareRec)
recode poverty (-2222.2=.), gen ( povertyRec)
recode dentist_rate (-2222.2=.), gen ( dentist_rateRec)
recode pap_smear (-2222.2=.) (-1111.1=.), gen ( Pap_SmearRate)
recode mammogram (-2222.2=.) (-1111.1=.), gen ( MammogrmRate)
recode proctoscopy (-2222.2=.) (-1111.1=.), gen ( ProctoscopRate)
recode unemployed (-9999=.) (-2222=.), gen (unemplRec)
recode sev_work_disabled (-2222=.), gen (sev_work_disablRec)
recode medianincome (-1=.), gen (MedianInc)
recode uninsured (-2222=.), gen (uninsuredRec)

	/*Recode for usability*/

recode community_health_center_ind (1=0) (2=1), gen (CenterYN)
recode hpsa_ind (1=0) (2=1), gen (HP_ShortageYN)
recode hpsa_ind (1=1) (2=0), gen (HP_ShortageY0N1)


	/*Generate new rate variables*/

*These variables were counts that needed to be turned into rates*
gen UNINSURE_Rate= (uninsuredRec/ population_size)
gen Elder_Medcar_Rate= ( elder_medicareRec/ population_size)
gen Disabl_Medcar_Rate= (  disabl_medicareRec/ population_size)
gen UNEMPL_Rate= (unemplRec/ population_size)
gen MAJ_Depress_Rate= (major_depression/ population_size)
gen SEV_Work_Disabl_Rate= (sev_work_disablRec/ population_size)

	/*These variables need to be turned into more wieldy rates*/

gen PHYS_Rateper100= (prim_care_phys_rate/ 1000)
gen DENT_Rateper100= (dentist_rateRec/ 1000)
gen EDU_AveYears= ( (18*EDUcollDegreeprcnt/100) + (13.5* EDUsomeCollegeprcnt/100) + (12* EDUonlyHSprcnt/100) + (10* EDUlessthanHSprcnt/100))
gen AGE_roughcompositete= ( (8* AGE19_64/100) + (38* AGE19_64/100) + (71* AGE65_84/100) + (88*  AGEover_85/100))
	/*Fianlly...*/

aorder
